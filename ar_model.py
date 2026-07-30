"""
BitDanceAR: GPT-style autoregressive transformer with binary diffusion head.

Replaces categorical next-token prediction (softmax over RVQ codes) with
flow-matching diffusion over continuous binary latent vectors {-1, +1}^C.

Key features:
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
from typing import Optional

from diff_head import DiffHead


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MAX_SPATIAL = 1024  # max Hq*Wq (supports up to 32×32 latent grids)
MAX_DIFF_H = 64             # max Hq for diffusion position embedding
MAX_DIFF_W = 64             # max Wq for diffusion position embedding


# ── Custom Architecture Components ──────────────────────────────────────────

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
    The MLP learns that row 0 is "top" and row Hq-1 is "bottom", etc.
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

    Each head gets its own SVD change-of-basis (A_h = U_h @ diag(s_h) @ V_h^T),
    aligning rotary planes with semantic directions while preserving RoPE's
    relative-offset property. Q is transformed by A, K by A^{-T}; the linear
    parts cancel in the dot product, but RoPE rotations operate in the mixed
    basis, effectively changing which dimension-pairs get rotated together.

    Supports KV caching for fast autoregressive inference. The cache stores
    K **after** A^{-T} and 2D-RoPE have been applied, so at generation time
    the new Q (after A + RoPE) attends to cached K directly.
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
        # A_h = U_h @ diag(s_h) @ V_h^T, where U_h, V_h are orthogonal
        # (constructed via matrix exponential of skew-symmetric matrices)
        # and s_h = softplus(sigma_h) > 0.
        num_skew = self.head_dim * (self.head_dim - 1) // 2
        self.U_skew = nn.Parameter(torch.zeros(n_head, num_skew))
        self.V_skew = nn.Parameter(torch.zeros(n_head, num_skew))
        # softplus(0.541) ≈ 1.0  →  initialize near identity
        self.sigma = nn.Parameter(torch.ones(n_head, self.head_dim) * 0.541)

        # 2D axial RoPE frequencies (half dims for x-axis, half for y-axis)
        theta = 10000.0 ** (-2 * torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        self.register_buffer('theta', theta, persistent=False)  # [half_dim/2]

        # KV cache state
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None
        self._cache_pos: int = 0

    # ── SVD helpers ──

    def _orth_from_skew(self, params: torch.Tensor) -> torch.Tensor:
        """Construct orthogonal matrices from skew-symmetric parameters via matrix exp.

        params: [H, num_skew] → [H, d, d] orthogonal matrices.
        Starting from zero params gives identity (matrix_exp(0) = I).
        """
        H = params.shape[0]
        d = self.head_dim
        idx = torch.triu_indices(d, d, offset=1, device=params.device)
        mats = torch.zeros(H, d, d, device=params.device, dtype=params.dtype)
        mats[:, idx[0], idx[1]] = params
        mats = mats - mats.transpose(-1, -2)  # skew-symmetric
        return torch.linalg.matrix_exp(mats)  # [H, d, d]

    def _apply_A(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply head-specific SVD change-of-basis.

        x: [B, H, T, d]
        Forward  (Q): x @ U @ diag(s)   @ V^T  = x @ A
        Inverse  (K): x @ U @ diag(1/s) @ V^T  = x @ A^{-T}

        In the attention dot product, the linear parts cancel:
            (x_q @ A)^T (x_k @ A^{-T}) = x_q^T x_k
        But RoPE is applied *after* the basis change, so the rotations
        operate in the mixed basis — this is the key HARoPE effect.
        """
        U = self._orth_from_skew(self.U_skew)  # [H, d, d]
        V = self._orth_from_skew(self.V_skew)  # [H, d, d]
        s = F.softplus(self.sigma)              # [H, d], guaranteed > 0

        # x @ U
        x = torch.einsum('bhtd,hde->bhte', x, U)
        if not inverse:
            x = x * s[None, :, None, :]          # diag(s)
        else:
            x = x / s[None, :, None, :]          # diag(1/s)
        x = torch.einsum('bhtd,hde->bhte', x, V.transpose(-1, -2))
        return x

    # ── 2D RoPE ──

    def _apply_2d_rope(self, x: torch.Tensor,
                       pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """Apply 2D axial RoPE.

        First half of dims are rotated by x-position, second half by y-position.
        This factorizes 2D position into two independent 1D RoPE blocks.

        x: [B, H, T, d]
        pos_x, pos_y: [B, T] — 2D coordinates for each token
        """
        # Expand pos for broadcasting across heads: [B, T] → [B, 1, T]
        px = pos_x.unsqueeze(1)  # [B, 1, T]
        py = pos_y.unsqueeze(1)  # [B, 1, T]

        x_x = x[..., :self.half_dim]    # x-axis dims
        x_y = x[..., self.half_dim:]    # y-axis dims

        x_x = self._apply_1d_rope(x_x, px)
        x_y = self._apply_1d_rope(x_y, py)
        return torch.cat([x_x, x_y], dim=-1)

    def _apply_1d_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply 1D RoPE to paired (even, odd) dimensions.

        x:   [B, H, T, half_dim]
        pos: [B, 1, T]  (broadcastable)
        """
        x1 = x[..., 0::2]  # [B, H, T, half_dim/2]
        x2 = x[..., 1::2]  # [B, H, T, half_dim/2]
        # pos: [B, 1, T] → [B, 1, T, 1] for broadcasting with theta [half_dim/2]
        angles = pos.unsqueeze(-1) * self.theta  # [B, 1, T, half_dim/2]
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
                pos_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_head
        hd = self.head_dim

        Q = self.q_norm((x @ self.Wq).reshape(B, T, H, hd).transpose(1, 2))
        K = self.k_norm((x @ self.Wk).reshape(B, T, H, hd).transpose(1, 2))
        V = (x @ self.Wv).reshape(B, T, H, hd).transpose(1, 2)

        # ── HARoPE: per-head SVD change-of-basis + 2D rotary ──
        if pos_x is not None and pos_y is not None:
            Q = self._apply_A(Q, inverse=False)   # A  for Q
            K = self._apply_A(K, inverse=True)    # A^{-T} for K
            Q = self._apply_2d_rope(Q, pos_x, pos_y)
            K = self._apply_2d_rope(K, pos_x, pos_y)

        # ── KV cache logic (stores K after A^{-T} + RoPE) ──
        if self._cache_k is not None:
            pos = self._cache_pos
            self._cache_k[:B, :, pos:pos + T] = K
            self._cache_v[:B, :, pos:pos + T] = V
            K_full = self._cache_k[:B, :, :pos + T]
            V_full = self._cache_v[:B, :, :pos + T]
            self._cache_pos = pos + T

            if T == 1:
                # Single-token generation: attend to all cached KVs
                Y = F.scaled_dot_product_attention(Q, K_full, V_full)
            elif pos == 0:
                # Initial prefill: Q and K same length, efficient causal
                Y = F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=True)
            else:
                # Multi-token with existing cache: custom causal mask
                q_idx = torch.arange(pos, pos + T, device=x.device)
                k_idx = torch.arange(pos + T, device=x.device)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                Y = F.scaled_dot_product_attention(Q, K_full, V_full,
                                                   attn_mask=causal_mask)
        else:
            # Training: standard causal attention
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # XSA: remove self-value component
        Vn = F.normalize(V, dim=-1)
        Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        out = Z.transpose(1, 2).reshape(B, T, D) @ self.Wo
        return out


# ── Transformer Block ───────────────────────────────────────────────────────

class GPTBlock(nn.Module):
    """Pre-norm causal GPT block with XSA + HARoPE and ReLU²."""
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
                pos_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), pos_x=pos_x, pos_y=pos_y)
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

    Architecture:
        [PAD_h, PAD_w, tag_1, ..., tag_N, SEP] → Causal Transformer → h[0..L-1]
        For each spatial position i: h[i] → DiffHead → pred_latent → sign() → {-1,+1}^C

    Position encoding:
        - Transformer attention: HARoPE (2D axial RoPE with per-head SVD basis)
        - DiffHead MLP: factorized 2D coordinate embedding (row + col)

    Configurable patch_size:
        patch_size=1: predict 1 spatial position per AR step (1x)
        patch_size=2: predict 2 spatial positions per AR step (2x)
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

        # Feature dim per AR position
        self.patch_dim = latent_dim * patch_size  # C * patch_size

        # Pad vocab to multiples of 64 for Tensor Core efficiency
        padded_tag_vocab = (tag_vocab_size + 63) // 64 * 64

        # ── Embeddings (no 1D pos_embed — HARoPE handles positions in attention) ──
        self.tag_embed = nn.Embedding(padded_tag_vocab, d_model)
        self.shape_h_embed = nn.Embedding(num_shape_buckets, d_model)
        self.shape_w_embed = nn.Embedding(num_shape_buckets, d_model)

        # ── Vision latent projector ──
        self.proj_in = MLPConnector(self.patch_dim, d_model, dropout)
        self.emb_norm = RMSNormNoParams()

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
              f"diff_layers={diff_layers}, diff_batch_mul={diff_batch_mul})")

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

        Returns:
            pos_x, pos_y: [B, N_prefix + n_img_tokens] float tensors
        """
        T = N_prefix + n_img_tokens
        pos_x = torch.zeros(B, T, device=device, dtype=torch.float32)
        pos_y = torch.zeros(B, T, device=device, dtype=torch.float32)

        # Prefix: 1D line at x=0, y = 0..N_prefix-1
        if N_prefix > 0:
            pos_y[:, :N_prefix] = torch.arange(
                N_prefix, device=device, dtype=torch.float32
            )

        # Image tokens: 2D grid positions
        if n_img_tokens > 0:
            p = self.patch_size
            patch_indices = torch.arange(
                start_patch, start_patch + n_img_tokens, device=device
            )
            spatial = patch_indices * p + p // 2  # center of patch
            rows = (spatial // Wq).float()
            cols = (spatial % Wq).float()
            pos_x[:, N_prefix:] = cols
            pos_y[:, N_prefix:] = rows

        return pos_x, pos_y

    def _diff_position_indices(self, L: int, Hq: int, Wq: int,
                               device: torch.device):
        """Compute (rows, cols) LongTensors for L patches (0..L-1).

        Used for the DiffusionPosEmbed which requires integer indices.
        """
        p = self.patch_size
        patch_indices = torch.arange(L, device=device, dtype=torch.long)
        spatial = patch_indices * p + p // 2
        rows = spatial // Wq
        cols = spatial % Wq
        return rows, cols

    # ── Patchify / Unpatchify ────────────────────────────────────────────────

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """
        [B, C, Hq, Wq] → [B, L, C*patch_size] where L = (Hq*Wq) // patch_size.
        Flattens spatial dims in raster order, groups adjacent positions.
        """
        B, C, Hq, Wq = latents.shape
        x = latents.reshape(B, C, -1).transpose(1, 2)  # [B, Hq*Wq, C]
        p = self.patch_size
        if p > 1:
            L = x.shape[1]
            assert L % p == 0, f"Spatial size {L} not divisible by patch_size {p}"
            x = x.reshape(B, L // p, C * p)
        return x  # [B, L, C*p]

    def unpatchify(self, x: torch.Tensor, Hq: int, Wq: int) -> torch.Tensor:
        """
        [B, L, C*patch_size] → [B, C, Hq, Wq]
        """
        B = x.shape[0]
        p = self.patch_size
        C = self.latent_dim
        if p > 1:
            x = x.reshape(B, -1, C)  # [B, Hq*Wq, C]
        x = x[:, :Hq * Wq, :]  # safety trim
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

    # ── Training Forward ─────────────────────────────────────────────────────

    def forward(
        self,
        tag_tokens: torch.Tensor,
        latents: torch.Tensor,
        shape_h_ids: torch.Tensor,
        shape_w_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            tag_tokens:   [B, N_prefix] — [PAD, PAD, tag_1, ..., SEP, PAD...]
            latents:      [B, C, Hq, Wq] — binary {-1, +1}
            shape_h_ids:  [B] — Hq values
            shape_w_ids:  [B] — Wq values
        Returns:
            loss: scalar diffusion loss
        """
        B = tag_tokens.shape[0]
        N_prefix = tag_tokens.shape[1]
        _, _, Hq, Wq = latents.shape

        # Patchify latent
        lat_seq = self.patchify(latents)  # [B