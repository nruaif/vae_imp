import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import builtins
import lpips
import timm
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob

from model import QwenRVQAutoencoder
from dataset import WDSLoader
from config import Config

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
# ==================== Losses & Metrics ====================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=5e-4):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = (pred - target).mean()
        pred_scaled = (pred.float() + 1) / 2
        target_scaled = (target.float() + 1) / 2
        pred_scaled = torch.clamp(pred_scaled, min=1e-6, max=1.0 - 1e-6)
        with torch.cuda.amp.autocast(enabled=False):
            bce_loss = F.binary_cross_entropy(pred_scaled, target_scaled)
        return diff**2 + bce_loss.mean()

def projection_loss(proj_emb, target_emb):
    """Combined cosine similarity + L2 loss for DINO projection matching."""
    cos_loss = 1.0 - F.cosine_similarity(proj_emb, target_emb, dim=-1).mean()
    l2_loss = F.mse_loss(proj_emb, target_emb)
    return cos_loss + l2_loss * 0.1


def compute_psnr(pred, target):
    pred = (pred.float() + 1) / 2
    target = (target.float() + 1) / 2
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


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


# ==================== DINO Teacher ====================

def setup_dino_teacher(device):
    """Load frozen DINO ViT-Small teacher for projection loss."""
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0)
    dino = dino.to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False
    # ImageNet normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return dino, mean, std


@torch.no_grad()
def get_dino_embedding(dino, x, mean, std):
    """Extract DINO CLS token embedding from images in [-1, 1] range."""
    x_dino = (x.float() + 1) / 2
    if x_dino.shape[-1] != 224 or x_dino.shape[-2] != 224:
        x_dino = F.interpolate(x_dino, size=224, mode='bicubic', align_corners=False).clamp(0, 1)
    x_dino = (x_dino - mean) / std
    return dino(x_dino)


# ==================== Training ====================

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    torch.backends.cudnn.benchmark = True

    if rank != 0:
        builtins.print = lambda *args, **kwargs: None

    cfg = Config.from_yaml(config_path)
    print(f"Device: {device}, Rank: {rank}, World Size: {world_size}")

    if rank == 0:
        wandb.init(project=cfg.training.wandb_project, config=cfg.to_dict())

    wds_loader = WDSLoader(
        url=cfg.data.webdataset_url,
        csv_path=cfg.data.csv_path,
        image_size=cfg.training.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dataloader = wds_loader.make_loader()

    # Projection config
    proj_cfg = getattr(cfg, 'projection', None)
    proj_hidden = getattr(proj_cfg, 'hidden_dim', 512) if proj_cfg else 512
    proj_output = getattr(proj_cfg, 'output_dim', 384) if proj_cfg else 384
    proj_weight = getattr(proj_cfg, 'weight', 0.5) if proj_cfg else 0.5
    proj_enabled = getattr(proj_cfg, 'enabled', True) if proj_cfg else True

    # Model
    model = QwenRVQAutoencoder(
        f=cfg.model.f,
        d_enc=cfg.model.d_enc,
        d_dec=cfg.model.d_dec,
        num_groups=cfg.model.num_groups,
        channels_per_group=cfg.model.channels_per_group,
        use_masking=cfg.training.use_masking,
        use_quant=getattr(cfg.model, 'use_quant', False),
        dropout=getattr(cfg.model, 'dropout', 0.0),
        proj_hidden_dim=proj_hidden,
        proj_output_dim=proj_output,
    ).to(device).to(memory_format=torch.channels_last)

    # DINO teacher
    dino_teacher, dino_mean, dino_std = None, None, None
    if proj_enabled:
        dino_teacher, dino_mean, dino_std = setup_dino_teacher(device)
        print(f"DINO teacher loaded (ViT-S/16). Projection weight: {proj_weight}")

    # Optimizer
    opt_gen = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    global_step = 0
    if dino_teacher:
        dino_teacher.compile()
    
    # Losses
    charbonnier = CharbonnierLoss()
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # Load checkpoint
    if getattr(cfg.training, 'resume_from', None) and os.path.exists(cfg.training.resume_from):
        print(f"Resuming from {cfg.training.resume_from}")
        checkpoint = torch.load(cfg.training.resume_from, map_location=device, )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            try:
                if "optimizer_state_dict" in checkpoint:
                    opt_gen.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                print(f"Warning: Failed to load optimizer/scaler state (likely due to architecture change): {e}")
                
            global_step = checkpoint.get("global_step", 0)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint at step {global_step}.")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # LR scheduler
    def lr_lambda(step):
        if step < cfg.training.warmup_steps:
            return float(step) / float(max(1, cfg.training.warmup_steps))
        return 1.0

    scheduler_gen = LambdaLR(opt_gen, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1)
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    if rank == 0:
        pbar = tqdm(range(global_step, cfg.training.max_train_steps), desc="Steps", dynamic_ncols=True)
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

        x_real = batch[0].to(device, memory_format=torch.channels_last)

        dino_emb = None
        if proj_enabled and dino_teacher is not None:
            dino_emb = get_dino_embedding(dino_teacher, x_real, dino_mean, dino_std)

        opt_gen.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            x_rec, latent, masked_ids, proj = model(x_real)
            recon_loss = charbonnier(x_rec, x_real)
            perc_loss = lpips_fn(x_rec, x_real).mean()

            g_total = recon_loss + 0.1 * perc_loss

            proj_loss = torch.tensor(0.0, device=device)
            if dino_emb is not None:
                proj_loss = projection_loss(proj, dino_emb)
                g_total = g_total + proj_weight * proj_loss

        scaler.scale(g_total).backward()
        scaler.unscale_(opt_gen)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt_gen)
        scaler.update()
        scheduler_gen.step()

        global_step += 1

        # Logging
        if rank == 0:
            pbar.update(1)
            model_raw = model.module if is_ddp else model
            current_temp = model_raw.quant.temp.item()

            psnr_batch = compute_psnr(x_rec, x_real)

            postfix = {
                "rec": f"{recon_loss.item():.4f}",
                "perc": f"{perc_loss.item():.4f}",
                "proj": f"{proj_loss.item():.4f}",
                "psnr": f"{psnr_batch:.1f}",
                "temp": f"{current_temp:.4f}",
            }
            pbar.set_postfix(**postfix)

            if global_step % cfg.training.log_every_steps == 0:
                wandb_log = {
                    "train/recon_loss": recon_loss.item(),
                    "train/perceptual_loss": perc_loss.item(),
                    "train/projection_loss": proj_loss.item(),
                    "train/total_loss": g_total.item(),
                    "train/lr": opt_gen.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/temp": current_temp,
                    "train/psnr_batch": psnr_batch,
                }
                wandb.log(wandb_log, step=global_step)

        # Checkpointing & Visualization
        if global_step % cfg.training.save_image_every_steps == 0:
            if is_ddp:
                dist.barrier()

            if rank == 0:
                print("\nSampling and Saving Checkpoint...")
                model_to_save = model.module if is_ddp else model
                ckpt_state = {
                    "model_state_dict": model_to_save.state_dict(),
                    "global_step": global_step,
                    "config": cfg.to_dict(),
                    "optimizer_state_dict": opt_gen.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }

                ckpt_path = os.path.join(cfg.training.output_dir, f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(cfg.training.output_dir, cfg.training.max_checkpoints, rank)

                model.eval()
                with torch.no_grad():
                    n_viz = min(8, x_real.shape[0])
                    x1_sample = x_real[:n_viz]

                    # 1. Evaluate with full channels (no masking)
                    x_rec_full, _, _, _ = model(x1_sample)
                    psnr_full = compute_psnr(x_rec_full, x1_sample)
                    full_grid = make_grid(
                        torch.cat([x1_sample, x_rec_full], dim=0),
                        nrow=n_viz, normalize=True, value_range=(-1, 1)
                    )
                    wandb.log({
                        "eval/recon_full_channels": wandb.Image(
                            full_grid,
                            caption=f"Step {global_step} | Full Channels PSNR: {psnr_full:.2f} dB"
                        )
                    }, step=global_step)

                    # 2. Evaluate with exactly 4 groups kept
                    x_rec_4g, _, _, _ = model(x1_sample, num_to_keep=4)
                    psnr_4g = compute_psnr(x_rec_4g, x1_sample)
                    grid_4g = make_grid(
                        torch.cat([x1_sample, x_rec_4g], dim=0),
                        nrow=n_viz, normalize=True, value_range=(-1, 1)
                    )
                    wandb.log({
                        "eval/recon_4_groups": wandb.Image(
                            grid_4g,
                            caption=f"Step {global_step} | 4 Groups PSNR: {psnr_4g:.2f} dB"
                        )
                    }, step=global_step)

                model.train()
                print("Done.\n")

            if is_ddp:
                dist.barrier()

    print("Training Complete.")
    if rank == 0 and pbar is not None:
        pbar.close()
        final_path = os.path.join(cfg.training.output_dir, 'ae_model_final.pth')
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)