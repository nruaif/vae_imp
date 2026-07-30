"""
BitDanceAR: GPT-style autoregressive transformer with binary diffusion head.

Key features:
- Dual-forward training: style extraction (full attention) + image generation (causal)
- 32 learnable style tokens for residual style prediction
- XSA (Exclusive Self-Attention) + HARoPE (Head-Aware Rotary Position Encoding)
  + ReLU² MLP blocks
- DiffHead for per-position flow-matching generation with 2D coordinate awareness
- KV cache for fast autoregressive inference
- Configurable patch_size (1x = 1 token/step, 2x = 2 tokens/step)
- Progressive group masking in DiffHead
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from diff_head import DiffHead
from sampling import euler_maruyama


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MAX_SPATIAL = 1024  # max Hq*Wq (supports up to 32×32 latent grids)
MAX_DIFF_H = 64             # max Hq for diffusion position embedding
MAX_DIFF_W = 64             # max Wq for diffusion position embedding
NUM_STYLE_TOKENS = 32       # number of learnable style tokens


# ── Custom Architecture Components ──────────────────────────────────────────

class BinarySTE(torch.autograd.Function):
    """Straight-Through Estimator for binary quantization. 
    Allows gradients to flow through the sign() function."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1] = 0
        return grad_input


class RMSNormNoParams(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms


class ReLUSquared(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2


class MLPConnector(nn.Module):
    """SwiGLU-style projector for mapping latent vectors into transformer dim."""
    def __init__(self, in_dim, dim, dropout_p=0.0):
        super().__init__()
        hidden_dim = int(dim * 1.5)
        self.w1 = nn.Linear(in_dim, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.ffn_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        h1, h2 = self.w1(x).chunk(2, dim=-1)
        return self.ffn_dropout(self.w2(F.silu(h1) * h2))


class DiffusionPosEmbed(nn.Module):
    """2D absolute coordinate encoding for MLP diffusion heads.

    Factorized row + column embeddings compose to give O(H+W) parameters
    instead of O(H*W), while preserving explicit 2D topology.
    """
    def __init__(self, d_model: int, max_h: int = MAX_DIFF_H,
                 max_w: int = MAX_DIFF_W):
        super().__init__()
        self.max_h = max_h
        self.max_w = max_w
        self.row_embed = nn.Embedding(max_h, d_model)
        self.col_embed = nn.Embedding(max_w, d_model)

    def forward(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """rows, cols: integer LongTensors of identical shape → [..., d_model]."""
        rows = rows.clamp(0, self.max_h - 1)
        cols = cols.clamp(0, self.max_w - 1)
        return self.row_embed(rows) + self.col_embed(cols)


def flip_binary(tensor: torch.Tensor, p_max: float) -> torch.Tensor:
    """Randomly flip signs of binary {-1,+1} tensor for training regularization."""
    if p_max <= 0.0:
        return tensor
    r1 = torch.rand_like(tensor)
    r2 = torch.rand_like(tensor)
    flip_mask = r1 < p_max * r2
    multiplier = torch.where(flip_mask, -1.0, 1.0).to(tensor.dtype)
    return tensor * multiplier


# ── Attention with KV Cache + HARoPE ────────────────────────────────────────

class ExclusiveSelfAttention(nn.Module):
    """
    Multi-head self-attention with:
    - XSA (exclusive self-attention): removes the component of the output
      along each position's own value vector.
    - HARoPE (Head-Aware Rotary Position Encoding): per-head SVD change-of-basis
      + 2D axial RoPE for relative position awareness in attention.
    - Supports both causal and full (bidirectional) attention via full_attention flag.
    - KV cache for fast autoregressive inference.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.half_dim = self.head_dim // 2
        assert self.head_dim % 4 == 0, (
            f"head_dim must be divisible by 4 for 2D HARoPE, got {self.head_dim}"
        )

        # Standard projections
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))

        self.q_norm = RMSNormNoParams()
        self.k_norm = RMSNormNoParams()

        nn.init.normal_(self.Wq, std=0.02)
        nn.init.normal_(self.Wk, std=0.02)
        nn.init.normal_(self.Wv, std=0.02)
        nn.init.zeros_(self.Wo)

        # ── HARoPE: per-head SVD parameters ──
        num_skew = self.head_dim * (self.head_dim - 1) // 2
        self.U_skew = nn.Parameter(torch.zeros(n_head, num_skew))
        self.V_skew = nn.Parameter(torch.zeros(n_head, num_skew))
        self.sigma = nn.Parameter(torch.ones(n_head, self.head_dim) * 0.541)

        # 2D axial RoPE frequencies
        theta = 10000.0 ** (-2 * torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        self.register_buffer('theta', theta, persistent=False)

        # KV cache state
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None
        self._cache_pos: int = 0

    # ── SVD helpers ──

    def _orth_from_skew(self, params: torch.Tensor) -> torch.Tensor:
        """Construct orthogonal matrices from skew-symmetric parameters via matrix exp."""
        H = params.shape[0]
        d = self.head_dim
        idx = torch.triu_indices(d, d, offset=1, device=params.device)
        mats = torch.zeros(H, d, d, device=params.device, dtype=params.dtype)
        mats[:, idx[0], idx[1]] = params
        mats = mats - mats.transpose(-1, -2)
        return torch.linalg.matrix_exp(mats)

    def _apply_A(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply head-specific SVD change-of-basis."""
        U = self._orth_from_skew(self.U_skew)
        V = self._orth_from_skew(self.V_skew)
        s = F.softplus(self.sigma)

        x = torch.einsum('bhtd,hde->bhte', x, U)
        if not inverse:
            x = x * s[None, :, None, :]
        else:
            x = x / s[None, :, None, :]
        x = torch.einsum('bhtd,hde->bhte', x, V.transpose(-1, -2))
        return x

    # ── 2D RoPE ──

    def _apply_2d_rope(self, x: torch.Tensor,
                       pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """Apply 2D axial RoPE. First half dims rotated by x, second half by y."""
        px = pos_x.unsqueeze(1)
        py = pos_y.unsqueeze(1)

        x_x = x[..., :self.half_dim]
        x_y = x[..., self.half_dim:]

        x_x = self._apply_1d_rope(x_x, px)
        x_y = self._apply_1d_rope(x_y, py)
        return torch.cat([x_x, x_y], dim=-1)

    def _apply_1d_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply 1D RoPE to paired (even, odd) dimensions."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        angles = pos.unsqueeze(-1) * self.theta
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        rx1 = x1 * cos_a - x2 * sin_a
        rx2 = x1 * sin_a + x2 * cos_a
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    # ── Cache helpers ──

    def enable_cache(self, bsz: int, max_len: int, device: torch.device,
                     dtype: torch.dtype):
        self._cache_k = torch.zeros(bsz, self.n_head, max_len, self.head_dim,
                                    device=device, dtype=dtype)
        self._cache_v = torch.zeros(bsz, self.n_head, max_len, self.head_dim,
                                    device=device, dtype=dtype)
        self._cache_pos = 0

    def disable_cache(self):
        self._cache_k = None
        self._cache_v = None
        self._cache_pos = 0

    # ── Forward ──

    def forward(self, x: torch.Tensor,
                pos_x: Optional[torch.Tensor] = None,
                pos_y: Optional[torch.Tensor] = None,
                full_attention: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_head
        hd = self.head_dim

        Q = self.q_norm((x @ self.Wq).reshape(B, T, H, hd).transpose(1, 2))
        K = self.k_norm((x @ self.Wk).reshape(B, T, H, hd).transpose(1, 2))
        V = (x @ self.Wv).reshape(B, T, H, hd).transpose(1, 2)

        # ── HARoPE: per-head SVD change-of-basis + 2D rotary ──
        if pos_x is not None and pos_y is not None:
            Q = self._apply_A(Q, inverse=False)
            K = self._apply_A(K, inverse=True)
            Q = self._apply_2d_rope(Q, pos_x, pos_y)
            K = self._apply_2d_rope(K, pos_x, pos_y)

        # ── KV cache logic ──
        if self._cache_k is not None:
            pos = self._cache_pos
            self._cache_k[:B, :, pos:pos + T] = K
            self._cache_v[:B, :, pos:pos + T] = V
            K_full = self._cache_k[:B, :, :pos + T]
            V_full = self._cache_v[:B, :, :pos + T]
            self._cache_pos = pos + T

            if T == 1:
                Y = F.scaled_dot_product_attention(Q, K_full, V_full)
            elif pos == 0:
                Y = F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=True)
            else:
                q_idx = torch.arange(pos, pos + T, device=x.device)
                k_idx = torch.arange(pos + T, device=x.device)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                Y = F.scaled_dot_product_attention(Q, K_full, V_full,
                                                   attn_mask=causal_mask)
        else:
            # Training: causal or full attention
            Y = F.scaled_dot_product_attention(
                Q, K, V, is_causal=not full_attention
            )

        # XSA: remove self-value component
        Vn = F.normalize(V, dim=-1)
        Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        out = Z.transpose(1, 2).reshape(B, T, D) @ self.Wo
        return out


# ── Transformer Block ───────────────────────────────────────────────────────

class GPTBlock(nn.Module):
    """Pre-norm GPT block with XSA + HARoPE and ReLU²."""
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.ln_1 = RMSNormNoParams()
        self.attn = ExclusiveSelfAttention(d_model, n_head)

        self.ln_2 = RMSNormNoParams()
        self.mlp_c_fc = nn.Linear(d_model, d_model * 4)
        self.mlp_act = ReLUSquared()
        self.mlp_c_proj = nn.Linear(d_model * 4, d_model)
        nn.init.zeros_(self.mlp_c_proj.weight)
        if self.mlp_c_proj.bias is not None:
            nn.init.zeros_(self.mlp_c_proj.bias)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                pos_x: Optional[torch.Tensor] = None,
                pos_y: Optional[torch.Tensor] = None,
                full_attention: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), pos_x=pos_x, pos_y=pos_y,
                          full_attention=full_attention)
        m = self.mlp_c_fc(self.ln_2(x))
        m = self.mlp_act(m)
        m = self.mlp_c_proj(m)
        m = self.mlp_drop(m)
        x = x + m
        return x


# ── Main Model ──────────────────────────────────────────────────────────────

class BitDanceAR(nn.Module):
    """
    GPT-style AR transformer with binary diffusion head for image generation.

    Dual-forward training:
      Forward 1 (style extraction):
        [32 learnable style queries, image tokens] → FULL attention
        → predict 32 bit-quantized residual style tokens
        → style loss (detached from transformer, only updates style tokens)

      Forward 2 (image generation):
        [shape, tags, style_cond (detached, first k∈{1,4,8,16,32}), image tokens]
        → CAUSAL attention → diffusion loss (updates transformer)

    Architecture:
        [shape_h, shape_w, tag_1, ..., tag_N, style_1, ..., style_k,
         img_0, ..., img_{L-1}] → Causal Transformer → DiffHead per position

    Position encoding:
        - Transformer attention: HARoPE (2D axial RoPE with per-head SVD basis)
        - DiffHead MLP: factorized 2D coordinate embedding (row + col)
    """

    def __init__(
        self,
        tag_vocab_size: int,
        d_model: int = 768,
        n_head: int = 12,
        n_layer: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        latent_dim: int = 56,
        num_groups: int = 4,
        channels_per_group: int = 14,
        patch_size: int = 1,
        num_shape_buckets: int = 64,
        diff_layers: int = 6,
        diff_dim: int = 768,
        diff_adanln_layers: int = 2,
        diff_batch_mul: int = 4,
        class_dropout_prob: float = 0.1,
        perturb_rate: float = 0.0,
        time_schedule: str = 'logit_normal',
        time_shift: float = 1.0,
        P_std: float = 0.8,
        P_mean: float = -0.8,
        pad_tag_id: int = 0,
        grad_checkpointing: bool = False,
        style_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.latent_dim = latent_dim
        self.num_groups = num_groups
        self.channels_per_group = channels_per_group
        self.patch_size = patch_size
        self.diff_batch_mul = diff_batch_mul
        self.class_dropout_prob = class_dropout_prob
        self.perturb_rate = perturb_rate
        self.pad_tag_id = pad_tag_id
        self.grad_checkpointing = grad_checkpointing
        self.style_loss_weight = style_loss_weight

        # Feature dim per AR position
        self.patch_dim = latent_dim * patch_size  # C * patch_size

        # Pad vocab to multiples of 64 for Tensor Core efficiency
        padded_tag_vocab = (tag_vocab_size + 63) // 64 * 64

        # ── Embeddings ──
        self.tag_embed = nn.Embedding(padded_tag_vocab, d_model)
        self.shape_h_embed = nn.Embedding(num_shape_buckets, d_model)
        self.shape_w_embed = nn.Embedding(num_shape_buckets, d_model)

        # ── Vision latent projector ──
        self.proj_in = MLPConnector(self.patch_dim, d_model, dropout)
        self.emb_norm = RMSNormNoParams()

        # ── Style tokens: 32 learnable tokens in latent_dim space ──
        # Serve as both queries (forward 1 input) and targets (style loss)
        self.style_tokens = nn.Parameter(
            torch.zeros(NUM_STYLE_TOKENS, latent_dim)
        )
        nn.init.normal_(self.style_tokens, std=0.02)

        # Style token projections
        self.style_query_proj = MLPConnector(latent_dim, d_model, dropout)
        self.style_pred_proj = nn.Linear(d_model, latent_dim)
        self.style_cond_proj = MLPConnector(latent_dim, d_model, dropout)

        # ── Transformer ──
        self.layers = nn.ModuleList([
            GPTBlock(d_model, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = RMSNormNoParams()

        # ── Diffusion head with 2D coordinate embedding ──
        self.pos_for_diff = DiffusionPosEmbed(d_model, max_h=MAX_DIFF_H,
                                              max_w=MAX_DIFF_W)
        self.head = DiffHead(
            ch_target=self.patch_dim,
            ch_cond=d_model,
            ch_latent=diff_dim,
            depth_latent=diff_layers,
            depth_adanln=diff_adanln_layers,
            num_groups=num_groups,
            channels_per_group=channels_per_group * patch_size,
            grad_checkpointing=grad_checkpointing,
            time_shift=time_shift,
            time_schedule=time_schedule,
            P_std=P_std,
            P_mean=P_mean,
        )

        self.apply(self._init_weights)
        self.head.initialize_weights()

        print(f"BitDanceAR: {self.num_params:,} params "
              f"(d={d_model}, L={n_layer}, H={n_head}, "
              f"latent={latent_dim}, patch={patch_size}x, "
              f"diff_layers={diff_layers}, diff_batch_mul={diff_batch_mul}, "
              f"style_tokens={NUM_STYLE_TOKENS})")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── 2D Position Helpers ──────────────────────────────────────────────────

    def _build_2d_positions(self, B: int, N_prefix: int, n_img_tokens: int,
                            Hq: int, Wq: int, device: torch.device,
                            start_patch: int = 0):
        """Build (pos_x, pos_y) for a sequence of [prefix, image] tokens.

        Prefix tokens are laid out as a 1D line at x=0:
            (0, 0), (0, 1), ..., (0, N_prefix-1)

        Image tokens use the 2D grid coordinates of their patch center:
            (col_j, row_j) for patch j = start_patch .. start_patch+n_img_tokens-1
        """
        T = N_prefix + n_img_tokens
        pos_x = torch.zeros(B, T, device=device, dtype=torch.float32)
        pos_y = torch.zeros(B, T, device=device, dtype=torch.float32)

        if N_prefix > 0:
            pos_y[:, :N_prefix] = torch.arange(
                N_prefix, device=device, dtype=torch.float32
            )

        if n_img_tokens > 0:
            p = self.patch_size
            patch_indices = torch.arange(
                start_patch, start_patch + n_img_tokens, device=device
            )
            spatial = patch_indices * p + p // 2
            rows = (spatial // Wq).float()
            cols = (spatial % Wq).float()
            pos_x[:, N_prefix:] = cols
            pos_y[:, N_prefix:] = rows

        return pos_x, pos_y

    def _diff_position_indices(self, L: int, Hq: int, Wq: int,
                               device: torch.device):
        """Compute (rows, cols) LongTensors for L patches (0..L-1)."""
        p = self.patch_size
        patch_indices = torch.arange(L, device=device, dtype=torch.long)
        spatial = patch_indices * p + p // 2
        rows = spatial // Wq
        cols = spatial % Wq
        return rows, cols

    # ── Patchify / Unpatchify ────────────────────────────────────────────────

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, C, Hq, Wq] → [B, L, C*patch_size] where L = (Hq*Wq) // patch_size."""
        B, C, Hq, Wq = latents.shape
        x = latents.reshape(B, C, -1).transpose(1, 2)  # [B, Hq*Wq, C]
        p = self.patch_size
        if p > 1:
            L = x.shape[1]
            assert L % p == 0, f"Spatial size {L} not divisible by patch_size {p}"
            x = x.reshape(B, L // p, C * p)
        return x

    def unpatchify(self, x: torch.Tensor, Hq: int, Wq: int) -> torch.Tensor:
        """[B, L, C*patch_size] → [B, C, Hq, Wq]"""
        B = x.shape[0]
        p = self.patch_size
        C = self.latent_dim
        if p > 1:
            x = x.reshape(B, -1, C)
        x = x[:, :Hq * Wq, :]
        return x.transpose(1, 2).reshape(B, C, Hq, Wq)

    # ── Tag dropout for CFG ──────────────────────────────────────────────────

    def drop_tags(self, tag_tokens: torch.Tensor) -> torch.Tensor:
        """Replace all tags with pad_id for CFG training."""
        if self.class_dropout_prob > 0 and self.training:
            B = tag_tokens.shape[0]
            drop_mask = torch.rand(B, device=tag_tokens.device) < self.class_dropout_prob
            tag_tokens = tag_tokens.clone()
            tag_tokens[drop_mask] = self.pad_tag_id
        return tag_tokens

    # ── Forward 1: Style Extraction ──────────────────────────────────────────

    def forward_style(
        self,
        latents: torch.Tensor,
        shape_h_ids: torch.Tensor,
        shape_w_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward 1: Extract style tokens from image with FULL attention.

        Sequence: [32 learnable style queries, image tokens]
        Attention: FULL (bidirectional)
        Output: bit-quantized predicted style tokens at the 32 query positions

        Args:
            latents:      [B, C, Hq, Wq] — binary {-1, +1}
            shape_h_ids:  [B]
            shape_w_ids:  [B]
        Returns:
            style_preds:     [B, 32, latent_dim] — continuous predictions
            style_preds_bin: [B, 32, latent_dim] — bit-quantized {-1, +1} via STE
        """
        B = latents.shape[0]
        _, _, Hq, Wq = latents.shape
        N_style = NUM_STYLE_TOKENS

        # ── Project style tokens to d_model as queries ──
        style_queries = self.style_query_proj(self.style_tokens)  # [32, d_model]
        style_queries = style_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 32, d_model]

        # ── Patchify and project image latents ──
        lat_seq = self.patchify(latents)  # [B, L, C*p]
        img_tokens = self.proj_in(lat_seq)  # [B, L, d_model]

        # ── Concatenate: [style_queries, img_tokens] ──
        L = img_tokens.shape[1]
        x = torch.cat([style_queries, img_tokens], dim=1)  # [B, 32+L, d_model]
        x = self.emb_norm(x)

        # ── Build 2D positions ──
        pos_x, pos_y = self._build_2d_positions(B, N_style, L, Hq, Wq, latents.device)

        # ── Run transformer with FULL attention ──
        for layer in self.layers:
            x = layer(x, pos_x=pos_x, pos_y=pos_y, full_attention=True)
        x = self.ln_f(x)

        # ── Extract style predictions at query positions ──
        style_preds = self.style_pred_proj(x[:, :N_style])  # [B, 32, latent_dim]

        # ── Bit quantize using STE so diff_loss can backprop through it ──
        style_preds_bin = BinarySTE.apply(style_preds)  # {-1, +1}

        return style_preds, style_preds_bin

    # ── Training Forward (Dual-Forward) ──────────────────────────────────────

    def forward(
        self,
        tag_tokens: torch.Tensor,
        latents: torch.Tensor,
        shape_h_ids: torch.Tensor,
        shape_w_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with dual-forward style + diffusion objectives.

        Forward 1: Style extraction (full attention)
            [32 learnable tokens, image] → predict style tokens → style loss
            Style loss is DETACHED from transformer — only updates style_tokens

        Forward 2: Image generation (causal attention)
            [shape, tags, style_cond, image] → diffusion loss
            Diffusion loss updates the transformer (and backprops through style_cond)

        Args:
            tag_tokens:   [B, N_prefix] — [PAD, PAD, tag_1, ..., SEP, PAD...]
            latents:      [B, C, Hq, Wq] — binary {-1, +1}
            shape_h_ids:  [B]
            shape_w_ids:  [B]
        Returns:
            total_loss:   scalar — diff_loss + style_loss_weight * style_loss
            diff_loss:    scalar — diffusion loss (updates transformer & Forward 1)
            style_loss:   scalar — style prediction loss (updates style_tokens only)
        """
        B = tag_tokens.shape[0]
        N_tags = tag_tokens.shape[1]
        _, _, Hq, Wq = latents.shape
        device = latents.device

        # ════════════════════════════════════════════════════════════════════
        # Forward 1: Style Extraction (FULL attention)
        # ════════════════════════════════════════════════════════════════════
        _, style_preds_bin = self.forward_style(latents, shape_h_ids, shape_w_ids)
        # style_preds_bin: [B, 32, latent_dim] in {-1, +1} via STE

        # Style loss: DETACHED. Only updates self.style_tokens, NO gradient to transformer
        style_target = self.style_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 32, latent_dim]
        style_loss = F.mse_loss(style_preds_bin.detach(), style_target)

        # ════════════════════════════════════════════════════════════════════
        # Forward 2: Image Generation (CAUSAL attention)
        # ════════════════════════════════════════════════════════════════════

        # ── Choose random k ∈ {1, 4, 8, 16, 32} style tokens to use ──
        k_choices = [1, 4, 8, 16, 32]
        k = k_choices[torch.randint(0, len(k_choices), (1,), device=device).item()]
        k = min(k, NUM_STYLE_TOKENS)

        # NO detach() here! We WANT the diff_loss to backprop through style_preds_bin
        # into the transformer (Forward 1) and the learnable style tokens.
        style_cond = style_preds_bin[:, :k]  # [B, k, latent_dim]
        style_cond_embed = self.style_cond_proj(style_cond)  # [B, k, d_model]

        # ── Build prefix: [shape_h, shape_w, tags, style_cond] ──
        shape_h = self.shape_h_embed(shape_h_ids).unsqueeze(1)  # [B, 1, d_model]
        shape_w = self.shape_w_embed(shape_w_ids).unsqueeze(1)  # [B, 1, d_model]
        tag_embeds = self.tag_embed(self.drop_tags(tag_tokens))  # [B, N_tags, d_model]

        prefix = torch.cat([shape_h, shape_w, tag_embeds, style_cond_embed], dim=1)
        N_prefix = prefix.shape[1]  # 2 + N_tags + k

        # ── Patchify and project image latents ──
        lat_seq = self.patchify(latents)  # [B, L, C*p] — clean targets
        L = lat_seq.shape[1]

        # Perturb image tokens for transformer input regularization
        if self.perturb_rate > 0 and self.training:
            lat_seq_perturbed = flip_binary(lat_seq, self.perturb_rate)
        else:
            lat_seq_perturbed = lat_seq

        img_tokens = self.proj_in(lat_seq_perturbed)  # [B, L, d_model]

        # ── Concatenate: [prefix, img_tokens] ──
        x = torch.cat([prefix, img_tokens], dim=1)  # [B, N_prefix + L, d_model]
        x = self.emb_norm(x)

        # ── Build 2D positions ──
        pos_x, pos_y = self._build_2d_positions(B, N_prefix, L, Hq, Wq, device)

        # ── Run transformer with CAUSAL attention ──
        for layer in self.layers:
            x = layer(x, pos_x=pos_x, pos_y=pos_y, full_attention=False)
        x = self.ln_f(x)

        # ── Extract conditioning hidden states for image token prediction ──
        # Hidden state at position (N_prefix-1+i) predicts image token i
        # So conditioning = x[:, N_prefix-1 : N_prefix-1+L]
        cond_hidden = x[:, N_prefix - 1: N_prefix - 1 + L]  # [B, L, d_model]

        # ── Add 2D position embedding for target patches ──
        rows, cols = self._diff_position_indices(L, Hq, Wq, device)
        pos_emb = self.pos_for_diff(rows, cols)  # [L, d_model]
        cond = cond_hidden + pos_emb.unsqueeze(0)  # [B, L, d_model]

        # ── Prepare for DiffHead ──
        # Flatten batch and sequence dims
        cond_flat = cond.reshape(B * L, -1)  # [B*L, d_model]
        targets_flat = lat_seq.reshape(B * L, -1)  # [B*L, C*p] — clean targets

        # Apply diff_batch_mul for more diverse timestep sampling
        if self.diff_batch_mul > 1:
            cond_flat = cond_flat.repeat(self.diff_batch_mul, 1)
            targets_flat = targets_flat.repeat(self.diff_batch_mul, 1)

        # ── Diffusion loss ──
        diff_loss = self.head(targets_flat, cond_flat)

        # ── Total loss ──
        total_loss = diff_loss + self.style_loss_weight * style_loss

        return total_loss, diff_loss, style_loss

    # ── Inference Sampling ───────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        tag_tokens: torch.Tensor,
        shape_h_id: torch.Tensor,
        shape_w_id: torch.Tensor,
        Hq: int,
        Wq: int,
        num_sampling_steps: int = 50,
        cfg_scale: float = 1.5,
        num_style_tokens: int = NUM_STYLE_TOKENS,
        reference_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive sampling with style conditioning.

        Args:
            tag_tokens:         [1, N_tags] — conditioning tags
            shape_h_id:         [1] — Hq bucket id
            shape_w_id:         [1] — Wq bucket id
            Hq, Wq:             latent grid dimensions
            num_sampling_steps: diffusion steps per AR token
            cfg_scale:          classifier-free guidance scale
            num_style_tokens:   number of style tokens to use (1-32)
            reference_latents:  optional [1, C, Hq_ref, Wq_ref] for style extraction
        Returns:
            latents: [1, C, Hq, Wq] — sampled binary latents in {-1, +1}
        """
        device = tag_tokens.device
        B = tag_tokens.shape[0]
        p = self.patch_size
        L = (Hq * Wq) // p
        k = min(num_style_tokens, NUM_STYLE_TOKENS)

        # ── Get style tokens ──
        if reference_latents is not None:
            # Extract style from reference image via forward 1
            _, style_preds_bin = self.forward_style(
                reference_latents, shape_h_id, shape_w_id
            )
            style_cond = style_preds_bin[:, :k]  # [B, k, latent_dim]
        else:
            # Use learnable style tokens (apply sign for binary)
            style_cond = torch.sign(self.style_tokens[:k])  # [k, latent_dim]
            style_cond = style_cond.unsqueeze(0).expand(B, -1, -1)  # [B, k, latent_dim]

        # ── CFG: duplicate batch (first half conditioned, second half unconditioned) ──
        use_cfg = cfg_scale != 1.0
        if use_cfg:
            B_eff = B * 2
            tag_tokens_eff = torch.cat([
                tag_tokens,
                torch.full_like(tag_tokens, self.pad_tag_id)
            ], dim=0)
            shape_h_eff = shape_h_id.repeat(2)
            shape_w_eff = shape_w_id.repeat(2)
            style_cond_eff = torch.cat([
                style_cond,
                torch.zeros_like(style_cond)
            ], dim=0)
        else:
            B_eff = B
            tag_tokens_eff = tag_tokens
            shape_h_eff = shape_h_id
            shape_w_eff = shape_w_id
            style_cond_eff = style_cond

        # ── Build prefix ──
        shape_h = self.shape_h_embed(shape_h_eff).unsqueeze(1)  # [B_eff, 1, d_model]
        shape_w = self.shape_w_embed(shape_w_eff).unsqueeze(1)  # [B_eff, 1, d_model]
        tag_embeds = self.tag_embed(tag_tokens_eff)  # [B_eff, N_tags, d_model]
        style_embeds = self.style_cond_proj(style_cond_eff)  # [B_eff, k, d_model]

        prefix = torch.cat([shape_h, shape_w, tag_embeds, style_embeds], dim=1)
        N_prefix = prefix.shape[1]
        prefix = self.emb_norm(prefix)

        # ── 2D positions for prefix ──
        pos_x_prefix, pos_y_prefix = self._build_2d_positions(
            B_eff, N_prefix, 0, Hq, Wq, device
        )

        # ── Enable KV cache ──
        max_len = N_prefix + L
        for layer in self.layers:
            layer.attn.enable_cache(B_eff, max_len, device, prefix.dtype)

        # ── Prefill prefix ──
        x = prefix
        for layer in self.layers:
            x = layer(x, pos_x=pos_x_prefix, pos_y=pos_y_prefix,
                      full_attention=False)
        x = self.ln_f(x)

        # Last prefix hidden state conditions first image token
        prev_hidden = x[:, -1:]  # [B_eff, 1, d_model]

        # ── Autoregressive generation ──
        sampled_tokens = []

        for i in range(L):
            # Target patch i's 2D position (for DiffHead position embedding)
            spatial = i * p + p // 2
            row_i = int(spatial // Wq)
            col_i = int(spatial % Wq)
            pos_emb = self.pos_for_diff(
                torch.tensor([row_i], device=device, dtype=torch.long),
                torch.tensor([col_i], device=device, dtype=torch.long),
            )  # [1, d_model]

            # Condition = prev_hidden + target position embedding
            cond = prev_hidden + pos_emb.unsqueeze(0)  # [B_eff, 1, d_model]

            # Apply CFG to hidden states
            if use_cfg:
                h_cond = cond[:B]      # [B, 1, d_model]
                h_uncond = cond[B:]    # [B, 1, d_model]
                h_cfg = h_uncond + cfg_scale * (h_cond - h_uncond)
                cond_flat = h_cfg.reshape(B, -1)  # [B, d_model]
            else:
                cond_flat = cond.reshape(B_eff, -1)  # [B, d_model]

            # Sample via diffusion head (CFG already applied to hidden states)
            sampled = self.head.sample(cond_flat, 1.0, num_sampling_steps)
            # sampled: [B, C*p]

            sampled_token = sampled.unsqueeze(1)  # [B, 1, C*p]
            sampled_tokens.append(sampled_token)

            # Project to d_model for next AR step
            next_input = self.proj_in(sampled_token)  # [B, 1, d_model]
            next_input = self.emb_norm(next_input)

            # Duplicate for cond and uncond paths if using CFG
            if use_cfg:
                next_input = next_input.repeat(2, 1, 1)  # [B_eff, 1, d_model]

            # HARoPE position for this token (patch i's 2D position)
            pos_x_next = torch.full(
                (B_eff, 1), float(col_i), device=device, dtype=torch.float32
            )
            pos_y_next = torch.full(
                (B_eff, 1), float(row_i), device=device, dtype=torch.float32
            )

            # Forward through transformer (single token, uses KV cache)
            x = next_input
            for layer in self.layers:
                x = layer(x, pos_x=pos_x_next, pos_y=pos_y_next,
                          full_attention=False)
            x = self.ln_f(x)

            prev_hidden = x  # [B_eff, 1, d_model]

        # ── Disable KV cache ──
        for layer in self.layers:
            layer.attn.disable_cache()

        # ── Collect and unpatchify ──
        all_tokens = torch.cat(sampled_tokens, dim=1)  # [B, L, C*p]
        latents = self.unpatchify(all_tokens, Hq, Wq)  # [B, C, Hq, Wq]
        return latents