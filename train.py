import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import builtins
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

# ==================== DINO Perceptual Loss (Reuses Teacher) ====================

class TimmDINOPerceptual(nn.Module):
    def __init__(self, teacher_model, target_size=224, layers="all", normalize=True):
        super().__init__()
        self.model = teacher_model
        self.target_size = target_size
        self.normalize = normalize
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_blocks = len(self.model.blocks)
        
        if layers is None or (isinstance(layers, str) and layers == 'all'):
            self.layers = list(range(self.num_blocks))
        elif isinstance(layers, int):
            self.layers = [layers]
        else:
            self.layers = list(layers)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() + 1.0) / 2.0
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                              mode='bicubic', align_corners=False).clamp(0, 1)
        return (x - self.mean) / self.std

    def _extract_features(self, x: torch.Tensor):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        
        outputs = {}
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.layers:
                outputs[i] = x
        return outputs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_proc = self._preprocess(pred)
        target_proc = self._preprocess(target).detach()

        feats_p = self._extract_features(pred_proc)
        feats_t = self._extract_features(target_proc)

        losses = []
        for i in self.layers:
            if i not in feats_p:
                continue
            fp = feats_p[i]
            ft = feats_t[i]
            
            if self.normalize:
                fp = F.normalize(fp, dim=-1)
                ft = F.normalize(ft, dim=-1)
            
            loss_i = (fp - ft).pow(2).mean(dim=(1, 2))
            losses.append(loss_i)

        if not losses:
            return torch.zeros(pred.shape[0], device=pred.device, dtype=torch.float32)
        
        return torch.stack(losses, dim=0).mean(dim=0)


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
        with torch.amp.autocast(device_type='cuda', enabled=False):
            bce_loss = F.binary_cross_entropy(pred_scaled, target_scaled)
        return diff**2 + bce_loss.mean()

def projection_loss(proj_emb, target_emb):
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
    if rank != 0: return
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        for ckpt in checkpoints[:-max_checkpoints]:
            try: os.remove(ckpt)
            except OSError: pass

# ==================== DINO Teacher ====================

def setup_dino_teacher(device):
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0)
    dino = dino.to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return dino, mean, std

@torch.no_grad()
def get_dino_features(dino, x, mean, std, target_size=14):
    x_dino = (x.float() + 1) / 2
    if x_dino.shape[-1] != 224 or x_dino.shape[-2] != 224:
        x_dino = F.interpolate(x_dino, size=224, mode='bicubic', align_corners=False).clamp(0, 1)
    x_dino = (x_dino - mean) / std
    feats = dino.forward_features(x_dino) # B, 197, 384
    
    cls = feats[:, 0]
    patches = feats[:, 1:]                # B, 196, 384
    
    # Interpolate DINO patches from 14x14 to match student's spatial grid
    B, N, C = patches.shape
    patches = patches.transpose(1, 2).view(B, C, 14, 14)
    if target_size != 14:
        patches = F.interpolate(patches, size=(target_size, target_size), mode='bilinear', align_corners=False)
    patches = patches.view(B, C, -1).transpose(1, 2) # B, target_size^2, 384
    
    return cls, patches

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
        image_size=cfg.training.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dataloader = wds_loader.make_loader()

    proj_cfg = getattr(cfg, 'projection', None)
    proj_weight = getattr(proj_cfg, 'weight', 0.5) if proj_cfg else 0.5
    proj_enabled = getattr(proj_cfg, 'enabled', True) if proj_cfg else True

    model = QwenRVQAutoencoder(
        f=cfg.model.f,
        d_enc=cfg.model.d_enc,
        d_dec=cfg.model.d_dec,
        num_groups=cfg.model.num_groups,
        channels_per_group=cfg.model.channels_per_group,
        use_masking=cfg.training.use_masking,
        use_quant=getattr(cfg.model, 'use_quant', False),
        dropout=getattr(cfg.model, 'dropout', 0.0),
    ).to(device).to(memory_format=torch.channels_last)

    dino_teacher, dino_mean, dino_std = None, None, None
    dino_perceptual = None
    if proj_enabled:
        dino_teacher, dino_mean, dino_std = setup_dino_teacher(device)
        dino_perceptual = TimmDINOPerceptual(
            dino_teacher,
            target_size=224,
            layers="all",
            normalize=True
        ).to(device)
        print(f"DINO teacher loaded (ViT-S/16). Projection weight: {proj_weight}")

    opt_gen = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
    
    global_step = 0
    charbonnier = CharbonnierLoss()

    # Correctly resume optimizer, model, and global_step
    if getattr(cfg.training, 'resume_from', None) and os.path.exists(cfg.training.resume_from):
        print(f"Resuming from {cfg.training.resume_from}")
        checkpoint = torch.load(cfg.training.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                opt_gen.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint.get("global_step", 0)
            print(f"Resumed successfully at step {global_step}.")
        else:
            model.load_state_dict(checkpoint, strict=False)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    def lr_lambda(step):
        if step < cfg.training.warmup_steps:
            return float(step) / float(max(1, cfg.training.warmup_steps))
        return 1.0

    # last_epoch is set to global_step - 1 to properly continue the LR schedule
    scheduler_gen = LambdaLR(opt_gen, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1)
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    if rank == 0:
        pbar = tqdm(range(global_step, cfg.training.max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)
    target_patch_size = cfg.training.image_size // cfg.model.f

    while global_step < cfg.training.max_train_steps:
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x_global = batch[0].to(device, memory_format=torch.channels_last)
        x_local = batch[1].to(device, memory_format=torch.channels_last)

        cls_global, patches_global, cls_local, patches_local = None, None, None, None
        if proj_enabled and dino_teacher is not None:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                cls_global, patches_global = get_dino_features(dino_teacher, x_global, dino_mean, dino_std, target_size=target_patch_size)
                cls_local, patches_local = get_dino_features(dino_teacher, x_local, dino_mean, dino_std, target_size=target_patch_size)

        opt_gen.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            img_global_rec_gm, img_local_rec_gm, img_local_rec_1d, \
            proj_patches_global, proj_patches_local, proj_summary = model(x_global, x_local)
            
            # Recon Losses
            recon_global_gm = charbonnier(img_global_rec_gm, x_global)
            recon_local_gm  = charbonnier(img_local_rec_gm, x_local)
            recon_local_1d  = charbonnier(img_local_rec_1d, x_local)
            
            # Perceptual Losses
            perc_global_gm = torch.tensor(0.0, device=device)
            perc_local_gm  = torch.tensor(0.0, device=device)
            perc_local_1d  = torch.tensor(0.0, device=device)
            if dino_perceptual is not None:
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    perc_global_gm = dino_perceptual(img_global_rec_gm, x_global).mean()
                    perc_local_gm  = dino_perceptual(img_local_rec_gm, x_local).mean()
                    perc_local_1d  = dino_perceptual(img_local_rec_1d, x_local).mean()

            g_total = recon_global_gm + recon_local_gm + recon_local_1d + \
                      500 * (perc_global_gm + perc_local_gm + perc_local_1d)

            # Proj Losses
            proj_loss_patches_g = torch.tensor(0.0, device=device)
            proj_loss_patches_l = torch.tensor(0.0, device=device)
            proj_loss_cls       = torch.tensor(0.0, device=device)
            
            if proj_enabled:
                proj_loss_patches_g = projection_loss(proj_patches_global.float(), patches_global)
                proj_loss_patches_l = projection_loss(proj_patches_local.float(), patches_local)
                proj_loss_cls = projection_loss(proj_summary.float(), cls_global)
                
                g_total = g_total + proj_weight * (proj_loss_patches_g + proj_loss_patches_l + proj_loss_cls)

        g_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_gen.step()
        scheduler_gen.step()

        global_step += 1

        if rank == 0:
            pbar.update(1)
            
            # Compute PSNR during training for logging
            psnr_train_g = compute_psnr(img_global_rec_gm.detach(), x_global)
            psnr_train_l_gm = compute_psnr(img_local_rec_gm.detach(), x_local)
            psnr_train_l_1d = compute_psnr(img_local_rec_1d.detach(), x_local)
            
            postfix = {
                "rec_g_gm": f"{recon_global_gm.item():.4f}",
                "rec_l_1d": f"{recon_local_1d.item():.4f}",
                "proj_cls": f"{proj_loss_cls.item():.4f}",
                "psnr_g":   f"{psnr_train_g:.2f}",
                "psnr_l_1d": f"{psnr_train_l_1d:.2f}",
            }
            pbar.set_postfix(**postfix)

            if global_step % cfg.training.log_every_steps == 0:
                wandb_log = {
                    "train/recon_global_gm": recon_global_gm.item(),
                    "train/recon_local_gm": recon_local_gm.item(),
                    "train/recon_local_1d": recon_local_1d.item(),
                    "train/perc_global_gm": perc_global_gm.item(),
                    "train/perc_local_1d": perc_local_1d.item(),
                    "train/proj_loss_patches_global": proj_loss_patches_g.item(),
                    "train/proj_loss_cls": proj_loss_cls.item(),
                    "train/total_loss": g_total.item(),
                    "train/lr": opt_gen.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/psnr_global_gm": psnr_train_g,
                    "train/psnr_local_gm": psnr_train_l_gm,
                    "train/psnr_local_1d": psnr_train_l_1d,
                }
                wandb.log(wandb_log, step=global_step)

        if global_step % cfg.training.save_image_every_steps == 0:
            if is_ddp: dist.barrier()

            if rank == 0:
                print("\nSampling and Saving Checkpoint...")
                model_to_save = model.module if is_ddp else model
                if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
                
                ckpt_state = {
                    "model_state_dict": model_to_save.state_dict(),
                    "global_step": global_step,
                    "config": cfg.to_dict(),
                    "optimizer_state_dict": opt_gen.state_dict(),
                }
                ckpt_path = os.path.join(cfg.training.output_dir, f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(cfg.training.output_dir, cfg.training.max_checkpoints, rank)

                model.eval()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    n_viz = min(4, x_global.shape[0])
                    x_g_viz = x_global[:n_viz]
                    x_l_viz = x_local[:n_viz]
                    
                    rec_g_gm, rec_l_gm, rec_l_1d, _, _, _ = model(x_g_viz, x_l_viz, eval_mask=True)
                    
                    psnr_g = compute_psnr(rec_g_gm, x_g_viz)
                    psnr_l_1d = compute_psnr(rec_l_1d, x_l_viz)
                    
                    grid_g = make_grid(torch.cat([x_g_viz, rec_g_gm], dim=0), nrow=n_viz, normalize=True, value_range=(-1, 1))
                    grid_l = make_grid(torch.cat([x_l_viz, rec_l_1d], dim=0), nrow=n_viz, normalize=True, value_range=(-1, 1))
                    
                    wandb.log({
                        "eval/global_gm_recon": wandb.Image(grid_g, caption=f"Step {global_step} | Global GM PSNR: {psnr_g:.2f} dB"),
                        "eval/local_1d_recon": wandb.Image(grid_l, caption=f"Step {global_step} | Local 1D PSNR: {psnr_l_1d:.2f} dB")
                    }, step=global_step)

                model.train()
                print("Done.\n")

            if is_ddp: dist.barrier()

    print("Training Complete.")
    if rank == 0 and pbar is not None:
        pbar.close()
        final_path = os.path.join(cfg.training.output_dir, 'ae_model_final.pth')
        model_to_save = model.module if is_ddp else model
        if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)