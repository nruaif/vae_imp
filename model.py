import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== Building Blocks ====================

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, dropout=0.0, layer_scale_init=1e-6):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)

        hidden_dim = in_ch * 4
        self.pwconv1 = nn.Linear(in_ch, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_ch)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(out_ch)) if layer_scale_init > 0 else None
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = x
        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
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
        h = self.unshuffle(x)
        h = self.channel_avg(h)
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
        x = self.pixel_unshuffle(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw_conv = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.pixel_shuffle(x)
        return x


# ==================== Qwen2 Encoder / Decoder ====================

class QwenImageVAE2Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=64, d_enc=96, n_layer=4, num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.n_layer = n_layer
        
        channels = [min(d_enc * (2 ** i), 768) for i in range(n_layer)]
        blocks_per_stage = [num_res_blocks * (2 ** i) for i in range(n_layer)][::-1]

        # Stem: 2x downsample
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        # Process initial resolution before first loop downsample
        self.initial_blocks = nn.ModuleList([
            ConvNeXtBlock(channels[0], channels[0], dropout=dropout)
            for _ in range(blocks_per_stage[0])
        ])

        self.downsamples = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        for i in range(n_layer - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            
            # Transition layers
            self.downsamples.append(Downsample(in_ch, out_ch))
            self.gscs.append(GlobalSkipConnection(in_ch, out_ch, downscale_factor=2))
            
            # Blocks operate safely on constant out_ch
            blocks = nn.ModuleList([
                ConvNeXtBlock(out_ch, out_ch, dropout=dropout) 
                for _ in range(blocks_per_stage[i+1])
            ])
            self.stages.append(blocks)
            
        self.norm_out = nn.BatchNorm2d(channels[-1])
        self.act = nn.SiLU()
        self.conv_to_latent = nn.Conv2d(channels[-1], latent_channels, 3, padding=1)

    # @torch.compile
    def forward(self, x):
        h = self.conv_in(x)
        
        for block in self.initial_blocks:
            h = block(h)
            
        for i in range(self.n_layer - 1):
            stage_input = h
            # Downsample first
            h_down = self.downsamples[i](h)
            gsc = self.gscs[i](stage_input, h_down.shape)
            h = h_down + gsc
            
            # Then process blocks
            for block in self.stages[i]:
                h = block(h)
            
        h = self.norm_out(h)
        h = self.act(h)
        return self.conv_to_latent(h)


class QwenImageVAE2Decoder(nn.Module):
    def __init__(self, latent_channels=64, out_channels=3, d_dec=144, n_layer=4, num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.n_layer = n_layer
        
        channels = []
        ch = d_dec
        for i in range(n_layer):
            channels.insert(0, min(ch, 1024))
            ch *= 2
            
        blocks_per_stage = [num_res_blocks * (2 ** i) for i in range(n_layer)][::-1]

        self.conv_from_latent = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        
        # Process latent resolution before first loop upsample
        self.initial_blocks = nn.ModuleList([
            ConvNeXtBlock(channels[0], channels[0], dropout=dropout)
            for _ in range(blocks_per_stage[0])
        ])
        
        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        for i in range(n_layer - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            
            # Transition layer
            self.upsamples.append(Upsample(in_ch, out_ch))

            # Blocks operate safely on constant out_ch
            blocks = nn.ModuleList([
                ConvNeXtBlock(out_ch, out_ch, dropout=dropout)
                for _ in range(blocks_per_stage[i+1])
            ])
            self.stages.append(blocks)
            
        self.norm_out = nn.BatchNorm2d(channels[-1])
        self.act = nn.SiLU()
        self.conv_out_full = nn.Sequential(
            nn.Conv2d(channels[-1], out_channels * 4, kernel_size=3, bias=True, padding=1,),
            nn.PixelShuffle(upscale_factor=2),
        )

    # @torch.compile
    def forward(self, z):
        h = self.conv_from_latent(z)
        
        for block in self.initial_blocks:
            h = block(h)
            
        for i in range(self.n_layer - 1):
            # Upsample first
            h = self.upsamples[i](h)
            
            # Then process blocks
            for block in self.stages[i]:
                h = block(h)
            
        h = self.norm_out(h)
        h = self.act(h)
        return self.conv_out_full(h)


# ==================== Quantizer & Proj Head ====================

class BinarySTE(torch.autograd.Function):
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

class AdaptiveBitwiseSign(nn.Module):
    def __init__(self, initial_temp=1.0):
        super().__init__()
        self.register_buffer('temp', torch.tensor(initial_temp, dtype=torch.float32))
    def forward(self, x):
        return BinarySTE.apply(x)
    def anneal_temp(self, factor=0.98, min_temp=0.01):
        self.temp.mul_(factor).clamp_(min=min_temp)

class ProjectionHead(nn.Module):
    def __init__(self, latent_channels, hidden_dim=512, output_dim=384):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        h = self.pool(z).flatten(1)
        return self.mlp(h)


# ==================== Main Model ====================

class QwenRVQAutoencoder(nn.Module):
    def __init__(self, f=16, d_enc=96, d_dec=144, num_groups=8, channels_per_group=14,
                 use_masking=True, use_quant=False, dropout=0.0,
                 proj_hidden_dim=512, proj_output_dim=384, num_res_blocks=2, **kwargs):
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
        self.quant = AdaptiveBitwiseSign()
        self.proj_head = ProjectionHead(self.latent_channels, proj_hidden_dim, proj_output_dim)

    def apply_group_masking(self, z, B, override_num_groups=None):
        # Only skip masking if we are evaluating AND not explicitly overriding groups
        if (not self.use_masking) or (not self.training and override_num_groups is None):
            return z, None
            
        z = z.clone()
        masked_ids = torch.arange(B, device=z.device)
        
        if override_num_groups is not None:
            # Inference/Testing: clamp to valid range
            n_keep = torch.full((B,), override_num_groups, dtype=torch.long, device=z.device)
            n_keep = torch.clamp(n_keep, 1, self.num_groups)
        else:
            # Training: Uniformly assign a random number of groups to keep for EVERY item in the batch
            n_keep = torch.randint(1, self.num_groups + 1, (B,), device=z.device)
            
        for i in range(B):
            keep = n_keep[i].item()
            start_ch = keep * self.channels_per_group
            if start_ch < self.latent_channels:
                z[i, start_ch:] = 0.0
                
        return z, masked_ids

    def encode(self, x):
        z = self.encoder(x)
        if self.use_quant:
            groups = z.chunk(self.num_groups, dim=1)
            groups = [self.quant(g) for g in groups]
            z = torch.cat(groups, dim=1)
        # Return 4 elements to match VQModel signature
        return z, None, None, None

    def decode(self, z):
        img = self.decoder(z)
        return torch.clamp(img, -1, 1)

    def forward(self, x, num_to_keep=None):
        z = self.encoder(x)
        B = z.shape[0]
        if self.use_quant:
            groups = z.chunk(self.num_groups, dim=1)
            groups = [self.quant(g) for g in groups]
            z = torch.cat(groups, dim=1)
        proj = self.proj_head(z)
        z, masked_ids = self.apply_group_masking(z, B, override_num_groups=num_to_keep)
        img = self.decoder(z)
        img = torch.clamp(img, -1, 1)
        return img, z, masked_ids, proj



# ==================== Quick Test ====================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenRVQAutoencoder(f=16, d_enc=96, d_dec=144).to(device)

    x = torch.randn(2, 3, 256, 256).to(device)
    img, z, masked_ids, proj = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {img.shape}")
    print(f"Proj shape:   {proj.shape}")
    print(f"Downsample factor: {x.shape[-1] // z.shape[-1]}x")

    assert x.shape == img.shape, f"Output shape mismatch: {img.shape} vs {x.shape}"
    assert x.shape[-1] // z.shape[-1] == 16, f"Expected 16x downsample, got {x.shape[-1] // z.shape[-1]}x"
    assert proj.shape == (2, 384), f"Projection shape mismatch: {proj.shape}"
    print("\nChecks passed! The architecture naturally downsamples/upsamples first.")
