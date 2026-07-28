# QwenRVQAutoencoder

Qwen-Image-VAE-2.0 backbone with grouped latent masking (RVQ-style).

## Architecture

- **Encoder**: Lightweight attention-free ResNet with Global Skip Connections (GSC)
- **Decoder**: Heavyweight attention-free ResNet (4 blocks/stage)
- **Latent**: Grouped channels (e.g. 8 groups × 14 ch = 112-dim latent)
- **Masking**: RVQ-style group dropping during training (later groups zeroed from the right)
- **No transformers, no CLS tokens, no KL loss**

## Quick Start

```bash
pip install -r requirements.txt

# Single GPU
python train.py --config config.yaml

# Multi-GPU (4 nodes)
torchrun --nproc_per_node=4 train.py --config config.yaml
```

## Data Format

Prepare WebDataset shards (`{00000..00004}.tar`) and a CSV with columns:
- `character`: class name string
- `id`: integer class id

Unknown characters are mapped to a null token (`num_classes`).

## Model Configs

| Name | f | Groups | Ch/Group | Latent Dim | Encoder | Decoder |
|------|---|--------|----------|------------|---------|---------|
| f16c112 | 16 | 8 | 14 | 112 | ~76M | ~248M |
| f16c96 | 16 | 8 | 12 | 96 | ~76M | ~248M |
| f32c128 | 32 | 8 | 16 | 128 | ~77M | ~250M |

Edit `config.yaml`:
```yaml
model:
  f: 16
  num_groups: 8
  channels_per_group: 14
```

## Training Features

- Charbonnier reconstruction loss + LPIPS perceptual loss
- Optional multi-scale PatchGAN with hinge loss & R1 penalty
- Discriminator warmup (autoencoder frozen initially)
- Mixed precision (bfloat16 generator, float16 discriminator)
- DDP support with gradient clipping
- Wandb logging with clean/masked reconstruction grids
- Automatic checkpoint rotation
