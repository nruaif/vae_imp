"""
Binary Diffusion Head with progressive group masking.
Adapted from BitDance (shallowdream204/BitDance).

Key idea: instead of categorical softmax over a codebook, we use a small
MLP-based flow-matching network to predict binary latent vectors.
Progressive group masking randomly selects how many of the 4 RVQ groups
to include in the loss, encouraging coarse-to-fine learning.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sampling import euler_maruyama


# ── Utilities ────────────────────────────────────────────────────────────────

def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1000.0):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    t = time_factor * t.float()
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def time_shift_sana(t, flow_shift=1., sigma=1.):
    return (1 / flow_shift) / ((1 / flow_shift) + (1 / t - 1) ** sigma)


# ── Diffusion Head ───────────────────────────────────────────────────────────

class DiffHead(nn.Module):
    """
    Flow-matching diffusion head with progressive group masking.
    
    Training: samples timestep t, interpolates target with noise,
    predicts clean data, computes masked MSE loss.
    
    Inference: runs Euler-Maruyama SDE to generate from noise.
    """

    def __init__(
        self,
        ch_target,
        ch_cond,
        ch_latent,
        depth_latent,
        depth_adanln,
        num_groups=4,
        channels_per_group=14,
        grad_checkpointing=False,
        time_shift=1.,
        time_schedule='logit_normal',
        P_std=1.,
        P_mean=0.,
    ):
        super().__init__()
        self.ch_target = ch_target
        self.time_shift = time_shift
        self.time_schedule = time_schedule
        self.P_std = P_std
        self.P_mean = P_mean
        self.num_groups = num_groups
        self.channels_per_group = channels_per_group

        self.net = MlpEncoder(
            in_channels=ch_target,
            model_channels=ch_latent,
            z_channels=ch_cond,
            num_res_blocks=depth_latent,
            num_ada_ln_blocks=depth_adanln,
            grad_checkpointing=grad_checkpointing,
        )

    def make_prefix_mask(self, c_prime, N, device):
        """
        Create per-sample channel masks selecting first c_prime groups.
        
        Args:
            c_prime: [N] tensor, each in {1, 2, 3, 4}
            N: batch size
            device: torch device
        Returns:
            mask: [N, C] binary mask
        """
        # c_prime * channels_per_group gives the cutoff index per sample
        cutoffs = (c_prime * self.channels_per_group).unsqueeze(1)  # [N, 1]
        indices = torch.arange(self.ch_target, device=device).unsqueeze(0)  # [1, C]
        mask = (indices < cutoffs).float()  # [N, C]
        return mask

    def forward(self, x, cond):
        """
        Compute flow-matching loss with per-sample progressive group masking.
        
        Args:
            x:    [N, C] target binary latent in {-1, +1}
            cond: [N, D] conditioning from AR transformer
        Returns:
            loss: scalar
        """
        N = x.shape[0]
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                # Per-sample progressive group masking
                c_prime = torch.randint(1, self.num_groups + 1, (N,), device=x.device)
                mask = self.make_prefix_mask(c_prime, N, x.device)

                # Sample timestep
                if self.time_schedule == 'logit_normal':
                    t = (torch.randn((N,), device=x.device) * self.P_std + self.P_mean).sigmoid()
                    if self.time_shift != 1.:
                        t = time_shift_sana(t, self.time_shift)
                elif self.time_schedule == 'uniform':
                    t = torch.rand((N,), device=x.device)
                    if self.time_shift != 1.:
                        t = time_shift_sana(t, self.time_shift)
                else:
                    raise NotImplementedError(f"unknown time_schedule {self.time_schedule}")

                # Flow interpolation: z_t = (1-t)*noise + t*data
                e = torch.randn_like(x)
                ti = t.view(-1, 1)
                z = (1.0 - ti) * e + ti * x

        # Network predicts clean data
        x_pred = self.net(z, t, cond)

        # Masked MSE loss (data prediction form)
        with torch.autocast(device_type="cuda", enabled=False):
            x_pred = x_pred.float()
            x = x.float()
            loss = F.mse_loss(x_pred * mask, x * mask)
        return loss

    def sample(self, z_cond, cfg, num_sampling_steps):
        """
        Sample via Euler-Maruyama SDE solver.
        
        Args:
            z_cond: [N, D] conditioning (doubled for CFG)
            cfg: CFG scale
            num_sampling_steps: number of solver steps
        Returns:
            samples: [N, ch_target]
        """
        return euler_maruyama(
            self.ch_target,
            self.net.forward,
            z_cond,
            cfg,
            num_sampling_steps=num_sampling_steps,
            time_shift=self.time_shift,
        )

    def initialize_weights(self):
        self.net.initialize_weights()


# ── Sub-modules ──────────────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ResBlock(nn.Module):
    """AdaLN-modulated residual MLP block with SwiGLU."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(self, x, scale, shift, gate):
        h = self.norm(x) * (1 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate


class FinalLayer(nn.Module):
    """AdaLN-modulated output projection."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.ada_ln_modulation = nn.Linear(channels, channels * 2, bias=True)
        self.linear = nn.Linear(channels, out_channels, bias=True)

    def forward(self, x, y):
        scale, shift = self.ada_ln_modulation(y).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        return self.linear(x)


class MlpEncoder(nn.Module):
    """
    AdaLN-modulated MLP encoder for diffusion head.
    Predicts clean data x from noisy input z, conditioned on timestep t
    and AR transformer hidden state c.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        z_channels,
        num_res_blocks,
        num_ada_ln_blocks=2,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])

        self.ada_ln_blocks = nn.ModuleList([
            nn.Linear(model_channels, model_channels * 3, bias=True)
            for _ in range(num_ada_ln_blocks)
        ])
        self.ada_ln_switch_freq = max(1, num_res_blocks // num_ada_ln_blocks)
        assert (num_res_blocks % self.ada_ln_switch_freq) == 0, \
            "num_res_blocks must be divisible by num_ada_ln_blocks"

        self.final_layer = FinalLayer(model_channels, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Args:
            x: [N, C] noisy input
            t: [N] timesteps in [0, 1]
            c: [N, D] conditioning from AR transformer
        Returns:
            [N, C] predicted clean data
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = F.silu(t + c)
        scale, shift, gate = self.ada_ln_blocks[0](y).chunk(3, dim=-1)

        if self.grad_checkpointing and self.training:
            for i, block in enumerate(self.res_blocks):
                if i > 0 and i % self.ada_ln_switch_freq == 0:
                    scale, shift, gate = self.ada_ln_blocks[i // self.ada_ln_switch_freq](y).chunk(3, dim=-1)
                x = checkpoint(block, x, scale, shift, gate, use_reentrant=False)
        else:
            for i, block in enumerate(self.res_blocks):
                if i > 0 and i % self.ada_ln_switch_freq == 0:
                    scale, shift, gate = self.ada_ln_blocks[i // self.ada_ln_switch_freq](y).chunk(3, dim=-1)
                x = block(x, scale, shift, gate)

        return self.final_layer(x, y)
