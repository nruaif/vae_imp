"""
BitDance AR Training Script.

Trains a BitDanceAR model on cached JSONL latents.
Uses flow-matching diffusion head with progressive group masking.
Includes periodic AR sampling/eval with optional VAE decoding.
Supports DDP for multi-GPU training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import builtins
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import wandb
import glob
import math
import numpy as np

from ar_model import BitDanceAR
from ar_dataset import make_ar_dataloader, TagVocab, LATENT_CHANNELS
from config import Config

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


# ==================== DDP Utilities ====================

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size, torch.device("cuda", local_rank)
    return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def cleanup_checkpoints(output_dir, max_checkpoints, rank):
    if rank != 0:
        return
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        for ckpt in checkpoints[:-max_checkpoints]:
            try:
                os.remove(ckpt)
            except OSError:
                pass


# ==================== VAE Decoding ====================

@torch.no_grad()
def decode_latents_to_image(latents, vae_model, device):
    """
    Decode binary latent tensor to an image via the VAE decoder.
    
    Args:
        latents: [C, Hq, Wq] binary tensor in {-1, +1}, C=56
        vae_model: QwenRVQAutoencoder instance
        device: torch device
    Returns:
        [3, H, W] image tensor in [-1, 1]
    """
    C, Hq, Wq = latents.shape
    
    # Pad to full VAE latent channels
    full_z = torch.zeros(1, vae_model.latent_channels, Hq, Wq, device=device)
    full_z[0, :C] = latents
    
    img = vae_model.decoder(full_z)
    return torch.clamp(img[0], -1, 1)


# ==================== LR Schedule ====================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr=1e-5, base_lr=3e-4):
    """Linear warmup then cosine decay to min_lr."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / base_lr
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ==================== AR Sampling / Eval ====================

@torch.no_grad()
def run_ar_eval(model_raw, vocab, batch, cfg, device, global_step, vae_model=None):
    """
    Run AR sampling on a few prompts and log results to WandB.
    Generates raw latent heatmaps and optionally VAE-decoded images.
    """
    import torchvision.utils as vutils

    model_raw.eval()
    num_samples = min(getattr(cfg.training, 'num_eval_samples', 4),
                      batch['tag_tokens'].shape[0])

    Hq = batch['shape_h_ids'][0].item()
    Wq = batch['shape_w_ids'][0].item()

    sample_steps = getattr(cfg.training, 'sample_steps', 50)
    sample_cfg = getattr(cfg.training, 'sample_cfg', 1.5)

    all_latent_grids = []
    all_decoded_imgs = []

    for b_idx in range(num_samples):
        tag_prefix = batch['tag_tokens'][b_idx:b_idx + 1].to(device)
        shape_h = batch['shape_h_ids'][b_idx:b_idx + 1].to(device)
        shape_w = batch['shape_w_ids'][b_idx:b_idx + 1].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                           enabled=device.type == "cuda"):
            sampled = model_raw.sample(
                tag_tokens=tag_prefix,
                shape_h_id=shape_h,
                shape_w_id=shape_w,
                Hq=Hq,
                Wq=Wq,
                num_sampling_steps=sample_steps,
                cfg_scale=sample_cfg,
            )  # [1, C, Hq, Wq]

        # Log raw latent heatmaps (show first 4 groups as grayscale)
        lat = sampled[0]  # [C, Hq, Wq]
        group_imgs = []
        for g in range(4):
            # Average the 14 binary channels per group
            ch_start = g * 14
            ch_end = (g + 1) * 14
            group_avg = lat[ch_start:ch_end].float().mean(dim=0, keepdim=True)  # [1, Hq, Wq]
            group_imgs.append((group_avg + 1) / 2)  # normalize to [0, 1]
        grid = vutils.make_grid(group_imgs, nrow=4, pad_value=1.0)
        all_latent_grids.append(grid.cpu())

        # VAE decoding
        if vae_model is not None:
            try:
                img = decode_latents_to_image(lat.to(device), vae_model, device)
                img_display = ((img.cpu() + 1) / 2).clamp(0, 1)
                all_decoded_imgs.append(img_display)
            except Exception as e:
                print(f"  VAE decode failed for sample {b_idx}: {e}")

    # Log to WandB
    for i, grid in enumerate(all_latent_grids):
        wandb.log({
            f"eval/latent_groups_{i}": wandb.Image(
                grid, caption=f"Step {global_step} | Sample {i} Latent Groups (G0-G3)"
            )
        }, step=global_step)

    for i, img in enumerate(all_decoded_imgs):
        wandb.log({
            f"eval/decoded_{i}": wandb.Image(
                img, caption=f"Step {global_step} | Sample {i} VAE Decoded"
            )
        }, step=global_step)

    if all_decoded_imgs:
        decoded_grid = vutils.make_grid(
            torch.stack(all_decoded_imgs), nrow=len(all_decoded_imgs),
            normalize=False, value_range=(0, 1)
        )
        wandb.log({
            "eval/decoded_grid": wandb.Image(
                decoded_grid, caption=f"Step {global_step} | All Decoded Samples"
            )
        }, step=global_step)


# ==================== Training ====================

def train(config_path: str):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    if rank != 0:
        builtins.print = lambda *args, **kwargs: None

    cfg = Config.from_yaml(config_path)
    print(f"Device: {device}, Rank: {rank}, World Size: {world_size}")

    if rank == 0:
        wandb.init(project=cfg.training.wandb_project, config=cfg.to_dict())

    # ── Data ──
    dataloader, vocab = make_ar_dataloader(
        cfg.data.data_dir,
        cfg.data.tag_csv,
        cfg.training.batch_size,
        cfg.training.num_workers,
        getattr(cfg.training, 'max_tags', 64),
    )
    print(f"Tag vocab size: {vocab.vocab_size}")

    # ── Model ──
    model = BitDanceAR(
        tag_vocab_size=vocab.vocab_size,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        max_seq_len=getattr(cfg.model, 'max_seq_len', 1280),
        dropout=getattr(cfg.model, 'dropout', 0.0),
        latent_dim=getattr(cfg.model, 'latent_dim', 56),
        num_groups=getattr(cfg.model, 'num_groups', 4),
        channels_per_group=getattr(cfg.model, 'channels_per_group', 14),
        patch_size=getattr(cfg.model, 'patch_size', 1),
        diff_layers=getattr(cfg.model, 'diff_layers', 6),
        diff_dim=getattr(cfg.model, 'diff_dim', 768),
        diff_adanln_layers=getattr(cfg.model, 'diff_adanln_layers', 2),
        diff_batch_mul=getattr(cfg.model, 'diff_batch_mul', 4),
        class_dropout_prob=getattr(cfg.model, 'class_dropout_prob', 0.1),
        perturb_rate=getattr(cfg.model, 'perturb_rate', 0.0),
        time_schedule=getattr(cfg.model, 'time_schedule', 'logit_normal'),
        time_shift=getattr(cfg.model, 'time_shift', 1.0),
        P_std=getattr(cfg.model, 'P_std', 0.8),
        P_mean=getattr(cfg.model, 'P_mean', -0.8),
        pad_tag_id=vocab.pad_id,
    ).to(device)

    # ── Optimizer (Muon + Adam) ──
    import muon

    hidden_matrix_params = []
    adam_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim >= 2 and "embed" not in n and "head" not in n:
                hidden_matrix_params.append(p)
            else:
                adam_params.append(p)

    base_lr = cfg.training.learning_rate
    muon_lr = base_lr * 100.0
    weight_decay = getattr(cfg.training, 'weight_decay', 0.05)

    adam_groups = [dict(params=adam_params, lr=base_lr, betas=(0.9, 0.95),
                        eps=1e-10, weight_decay=0, use_muon=False)]
    muon_group = dict(params=hidden_matrix_params, lr=muon_lr,
                      momentum=0.95, weight_decay=weight_decay, use_muon=True)
    param_groups = [*adam_groups, muon_group]

    if is_ddp:
        optimizer = muon.MuonWithAuxAdam(param_groups)
    else:
        optimizer = muon.SingleDeviceMuonWithAuxAdam(param_groups)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        cfg.training.warmup_steps,
        cfg.training.max_train_steps,
        min_lr=1e-5,
        base_lr=cfg.training.learning_rate,
    )

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    global_step = 0

    # ── Resume ──
    if getattr(cfg.training, 'resume_from', None) and os.path.exists(cfg.training.resume_from):
        print(f"Resuming from {cfg.training.resume_from}")
        ckpt = torch.load(cfg.training.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, cfg.training.warmup_steps, cfg.training.max_train_steps,
            min_lr=1e-5, base_lr=cfg.training.learning_rate,
        )
        scheduler.last_epoch = global_step
        print(f"Loaded checkpoint at step {global_step}.")

    # ── DDP ──
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Optional VAE for decoding samples ──
    vae_model = None
    if getattr(cfg, 'vae', None) and getattr(cfg.vae, 'checkpoint', None):
        try:
            from model import QwenRVQAutoencoder
            vae_model = QwenRVQAutoencoder(
                f=cfg.vae.f,
                d_enc=cfg.vae.d_enc,
                d_dec=cfg.vae.d_dec,
                num_groups=cfg.vae.num_groups,
                channels_per_group=cfg.vae.channels_per_group,
                use_quant=True,
            ).to(device)
            state = torch.load(cfg.vae.checkpoint, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                vae_model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                vae_model.load_state_dict(state, strict=False)
            vae_model.eval()
            for p in vae_model.parameters():
                p.requires_grad = False
            print(f"VAE loaded from {cfg.vae.checkpoint} for sample decoding")
        except Exception as e:
            print(f"Warning: failed to load VAE: {e}")
            vae_model = None

    os.makedirs(cfg.training.output_dir, exist_ok=True)

    # ── Training loop ──
    if rank == 0:
        pbar = tqdm(total=cfg.training.max_train_steps, initial=global_step,
                    desc="BitDance AR Training", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

    while global_step < cfg.training.max_train_steps:
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                           enabled=device.type == "cuda"):
            loss_fm, nttp = model(
                tag_tokens=batch['tag_tokens'],
                latents=batch['latents'],
                shape_h_ids=batch['shape_h_ids'],
                shape_w_ids=batch['shape_w_ids'],
            )
        loss = loss_fm 
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        # ── Logging ──
        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss_fm.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "nttp": f"{nttp.item():.4f}",
            })

            if global_step % cfg.training.log_every_steps == 0:
                log_dict = {
                    "train/loss": loss_fm.item(),
                    "train/nttp": f"{nttp.item():.4f}",

                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                }
                wandb.log(log_dict, step=global_step)

        # ── Checkpoint & Eval ──
        if global_step % cfg.training.save_every_steps == 0:
            if is_ddp:
                dist.barrier()

            if rank == 0:
                print(f"\n[Step {global_step}] Saving checkpoint & running eval...")
                model_raw = model.module if is_ddp else model

                # Save checkpoint
                ckpt_state = {
                    "model_state_dict": model_raw.state_dict(),
                    "global_step": global_step,
                    "config": cfg.to_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }
                ckpt_path = os.path.join(cfg.training.output_dir, f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(cfg.training.output_dir, cfg.training.max_checkpoints, rank)

                # AR sampling eval
                try:
                    run_ar_eval(model_raw, vocab, batch, cfg, device, global_step, vae_model)
                except Exception as e:
                    print(f"  Eval failed: {e}")

                model.train()
                print("Done.\n")

            if is_ddp:
                dist.barrier()

    # ── Finish ──
    print("Training complete.")
    if rank == 0 and pbar is not None:
        pbar.close()
        final_path = os.path.join(cfg.training.output_dir, 'ar_model_final.pth')
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train BitDance AR image model")
    parser.add_argument('--config', type=str, default='ar_config.yaml')
    args = parser.parse_args()
    train(args.config)
