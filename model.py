import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_ = torch._dynamo
torch._functorch.config.activation_memory_budget = 0.5

# ==================== Building Blocks ====================

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, dropout=0.0, layer_scale_init=1e-6):
        super().__init__()
        out_ch = out_ch or in_ch
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        hidden_dim = in_ch * 4
        self.pwconv1 = nn.Linear(in_ch, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(out_ch)) if layer_scale_init > 0 else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    @torch.compile
    def forward(self, x):
        residual = x
        h = self.dwconv(x).permute(0, 2, 3, 1)
        h = self.norm(h)
        h = self.act(self.pwconv1(h))
        h = self.drop(h)
        h = self.pwconv2(h)
        if self.gamma is not None:
            h = self.gamma * h
        h = h.permute(0, 3, 1, 2)
        return h + self.skip(residual)


class GlobalSkipConnection(nn.Module):
    def __init__(self, in_ch, out_ch, downscale_factor=2):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channel_avg = nn.Conv2d(in_ch * (downscale_factor ** 2), out_ch, 1, bias=False)

    def forward(self, x, target_shape):
        h = self.channel_avg(self.unshuffle(x))
        if h.shape[2:] != target_shape[2:]:
            h = F.interpolate(h, size=target_shape[2:], mode="bilinear", align_corners=False)
        return h


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.dw_conv = nn.Conv2d(in_ch * 4, in_ch * 4, 3, padding=1, groups=in_ch * 4)
        self.pw_conv = nn.Conv2d(in_ch * 4, out_ch, 1)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(self.pixel_unshuffle(x)))


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw_conv = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.pw_conv(self.dw_conv(x)))


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    @torch.compile
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== Encoder / Decoder ====================

class QwenImageVAE2Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=64, d_enc=96, n_layer=4,
                 num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.n_layer = n_layer
        channels = [min(d_enc * (2 ** i), 768) for i in range(n_layer)]
        blocks_per_stage = [num_res_blocks * (2 ** i) for i in range(n_layer)][::-1]

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]), nn.SiLU(),
        )
        self.initial_blocks = nn.ModuleList([
            ConvNeXtBlock(channels[0], channels[0], dropout=dropout)
            for _ in range(blocks_per_stage[0])
        ])
        self.downsamples = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(n_layer - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            self.downsamples.append(Downsample(in_ch, out_ch))
            self.gscs.append(GlobalSkipConnection(in_ch, out_ch, downscale_factor=2))
            self.stages.append(nn.ModuleList([
                ConvNeXtBlock(out_ch, out_ch, dropout=dropout)
                for _ in range(blocks_per_stage[i + 1])
            ]))
        self.norm_out = nn.BatchNorm2d(channels[-1])
        self.act = nn.SiLU()
        self.conv_to_latent = nn.Conv2d(channels[-1], latent_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.initial_blocks:
            h = block(h)
        for i in range(self.n_layer - 1):
            h_down = self.downsamples[i](h)
            gsc = self.gscs[i](h, h_down.shape)
            h = h_down + gsc
            for block in self.stages[i]:
                h = block(h)
        return self.conv_to_latent(self.act(self.norm_out(h)))


class QwenImageVAE2Decoder(nn.Module):
    def __init__(self, latent_channels=64, out_channels=3, d_dec=144, n_layer=4,
                 num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.n_layer = n_layer
        channels, ch = [], d_dec
        for _ in range(n_layer):
            channels.insert(0, min(ch, 1024))
            ch *= 2
        blocks_per_stage = [num_res_blocks * (2 ** i) for i in range(n_layer)][::-1]

        self.conv_from_latent = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        self.initial_blocks = nn.ModuleList([
            ConvNeXtBlock(channels[0], channels[0], dropout=dropout)
            for _ in range(blocks_per_stage[0])
        ])
        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(n_layer - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            self.upsamples.append(Upsample(in_ch, out_ch))
            self.stages.append(nn.ModuleList([
                ConvNeXtBlock(out_ch, out_ch, dropout=dropout)
                for _ in range(blocks_per_stage[i + 1])
            ]))
        self.norm_out = nn.BatchNorm2d(channels[-1])
        self.act = nn.SiLU()
        self.conv_out_full = nn.Sequential(
            nn.Conv2d(channels[-1], out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
        )

    def forward(self, z):
        h = self.conv_from_latent(z)
        for block in self.initial_blocks:
            h = block(h)
        for i in range(self.n_layer - 1):
            h = self.upsamples[i](h)
            for block in self.stages[i]:
                h = block(h)
        return self.conv_out_full(self.act(self.norm_out(h)))


# ==================== Quantizer & Proj Heads ====================

class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[x.abs() > 1] = 0
        return grad

class BinarySign(nn.Module):
    def forward(self, x):
        return BinarySTE.apply(x)

class ProjectionHeadPatches(nn.Module):
    def __init__(self, latent_channels, output_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, output_dim, 1),
            nn.GroupNorm(1, output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, 1)
        )
    def forward(self, z):
        h = self.net(z)
        return h.flatten(2).transpose(1, 2)

class ProjectionHeadSummary(nn.Module):
    def __init__(self, latent_channels, output_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_channels),
            nn.Linear(latent_channels, latent_channels * 2),
            nn.GELU(),
            nn.Linear(latent_channels * 2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


# ==================== Main Model ====================

class QwenRVQAutoencoder(nn.Module):
    def __init__(self, f=16, d_enc=96, d_dec=144, num_groups=8, channels_per_group=14,
                 use_masking=True, use_quant=False, dropout=0.0,
                 proj_hidden_dim=512, proj_output_dim=384, num_res_blocks=2,
                 num_summary_tokens=32, num_spatial_tokens=256, summary_bits=13,
                 summary_min_keep=4, **kwargs):
        super().__init__()
        self.num_groups = num_groups
        self.channels_per_group = channels_per_group
        self.latent_channels = num_groups * channels_per_group
        self.use_masking = use_masking
        self.use_quant = use_quant

        n_layer = int(math.log2(f))

        self.encoder = QwenImageVAE2Encoder(
            in_channels=3, latent_channels=self.latent_channels,
            d_enc=d_enc, n_layer=n_layer, num_res_blocks=num_res_blocks, dropout=dropout,
        )
        self.decoder = QwenImageVAE2Decoder(
            latent_channels=self.latent_channels, out_channels=3,
            d_dec=d_dec, n_layer=n_layer, num_res_blocks=num_res_blocks, dropout=dropout,
        )
        self.quant = BinarySign()
        
        self.proj_head_patches = ProjectionHeadPatches(self.latent_channels, proj_output_dim)
        self.proj_head_tok32 = ProjectionHeadSummary(self.latent_channels, proj_output_dim)

        self.num_summary_tokens = num_summary_tokens
        self.num_spatial_tokens = num_spatial_tokens
        self.summary_bits = summary_bits

        self.tokens_32 = nn.Parameter(torch.zeros(1, num_summary_tokens, self.latent_channels))
        self.pos_embed_1d = nn.Parameter(torch.zeros(1, num_spatial_tokens, self.latent_channels))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.latent_channels))
        for p in (self.tokens_32, self.pos_embed_1d, self.mask_token):
            nn.init.normal_(p, std=0.02)

        self.vit_enc_1d = nn.Sequential(*[
            ViTBlock(self.latent_channels, num_heads=8, dropout=dropout) for _ in range(8)
        ])
        self.vit_dec_1d = nn.Sequential(*[
            ViTBlock(self.latent_channels, num_heads=8, dropout=dropout) for _ in range(8)
        ])

        self.proj_to_bits = nn.Linear(num_summary_tokens * self.latent_channels, summary_bits)
        self.proj_from_bits = nn.Linear(summary_bits, num_summary_tokens * self.latent_channels)

    def apply_group_masking(self, z, B):
        if not self.use_masking:
            return z, None
        z = z.clone()
        n_keep = torch.randint(1, self.num_groups + 1, (B,), device=z.device)
        n_keep = torch.clamp(n_keep, 1, self.num_groups)

        group_idx = torch.arange(self.num_groups, device=z.device).unsqueeze(0)
        keep_groups = group_idx < n_keep.unsqueeze(1)
        keep_channels = keep_groups.repeat_interleave(self.channels_per_group, dim=1)
        z = z * keep_channels.unsqueeze(-1).unsqueeze(-1).to(z.dtype)
        return z, torch.arange(B, device=z.device)

    def apply_spatial_masking(self, spatial_tokens, mask_ratio=0.9):
        B, N, C = spatial_tokens.shape
        H = W = int(math.sqrt(N))
        block = 2
        num_blocks = (H // block) * (W // block)
        num_keep = int(num_blocks * (1 - mask_ratio))

        noise = torch.rand(B, num_blocks, device=spatial_tokens.device)
        ids_keep = torch.argsort(noise, dim=1)[:, :num_keep]

        r = ids_keep // (W // block)
        c = ids_keep % (W // block)
        t1 = r * (block * W) + c * block
        keep_ids = torch.cat([t1, t1 + 1, t1 + W, t1 + W + 1], dim=1)

        mask = torch.ones(B, N, device=spatial_tokens.device, dtype=torch.bool)
        mask.scatter_(1, keep_ids, False)
        return torch.where(mask.unsqueeze(-1),
                           self.mask_token.expand_as(spatial_tokens),
                           spatial_tokens)

    def forward(self, x_global, x_local, eval_mask=False):
        B = x_global.shape[0]

        # --- 1. CNN Encoder & Group Masking for BOTH crops ---
        z_global = self.encoder(x_global)
        z_local = self.encoder(x_local)

        if self.use_quant:
            z_global = torch.cat([self.quant(g) for g in z_global.chunk(self.num_groups, dim=1)], dim=1)
            z_local = torch.cat([self.quant(g) for g in z_local.chunk(self.num_groups, dim=1)], dim=1)

        z_global_masked, _ = self.apply_group_masking(z_global, B)
        z_local_masked, _ = self.apply_group_masking(z_local, B)

        img_global_rec_gm = torch.clamp(self.decoder(z_global_masked), -1, 1)
        img_local_rec_gm = torch.clamp(self.decoder(z_local_masked), -1, 1)

        proj_patches_global = self.proj_head_patches(z_global)
        proj_patches_local = self.proj_head_patches(z_local)

        # --- 2. Transformer Summary from Global ---
        spatial_global = z_global.flatten(2).transpose(1, 2)
        N_global = spatial_global.shape[1]
        H_global = int(math.sqrt(N_global))
        
        if N_global == self.pos_embed_1d.shape[1]:
            spatial_global = spatial_global + self.pos_embed_1d
        else:
            pe = self.pos_embed_1d.transpose(1, 2).view(1, self.latent_channels, int(math.sqrt(self.num_spatial_tokens)), int(math.sqrt(self.num_spatial_tokens)))
            pe = F.interpolate(pe, size=(H_global, H_global), mode='bicubic', align_corners=False)
            spatial_global = spatial_global + pe.flatten(2).transpose(1, 2)

        tok_32 = self.tokens_32.expand(B, -1, -1)
        enc_in = torch.cat([tok_32, spatial_global], dim=1)
        enc_out = self.vit_enc_1d(enc_in)
        tok_32_out = enc_out[:, :32]

        flat_32 = tok_32_out.flatten(1)
        quant_bits = self.quant(self.proj_to_bits(flat_32))
        tok_32_q = self.proj_from_bits(quant_bits).view(B, 32, self.latent_channels)

        # --- 3. Transformer Decoder on Masked Local ---
        spatial_local = z_local.flatten(2).transpose(1, 2)
        N_local = spatial_local.shape[1]
        H_local = int(math.sqrt(N_local))
        
        if N_local == self.pos_embed_1d.shape[1]:
            spatial_local = spatial_local + self.pos_embed_1d
        else:
            pe = self.pos_embed_1d.transpose(1, 2).view(1, self.latent_channels, int(math.sqrt(self.num_spatial_tokens)), int(math.sqrt(self.num_spatial_tokens)))
            pe = F.interpolate(pe, size=(H_local, H_local), mode='bicubic', align_corners=False)
            spatial_local = spatial_local + pe.flatten(2).transpose(1, 2)

        if self.training or eval_mask:
            masked_spatial_local = self.apply_spatial_masking(spatial_local, mask_ratio=0.9)
        else:
            masked_spatial_local = spatial_local

        dec_in = torch.cat([tok_32_q, masked_spatial_local], dim=1)
        dec_out = self.vit_dec_1d(dec_in)

        z_rec_local = dec_out[:, 32:].transpose(1, 2).view(B, self.latent_channels, H_local, H_local)
        img_local_rec_1d = torch.clamp(self.decoder(z_rec_local), -1, 1)

        proj_summary = self.proj_head_tok32(tok_32_out.mean(dim=1))

        return (img_global_rec_gm, img_local_rec_gm, img_local_rec_1d,
                proj_patches_global, proj_patches_local, proj_summary)