"""
train.py
--------
Pipeline huấn luyện với DistributedDataParallel (DDP) trên 2× T4 GPU.

Các thành phần chính:
  - setup_ddp / cleanup_ddp  : Khởi tạo và dọn dẹp process group
  - train_epoch              : 1 epoch training với AMP + tqdm
  - val_epoch                : 1 epoch validation với AMP + tqdm
  - train_worker             : Hàm worker cho mỗi rank (spawn target)
  - run_training             : Entry point DDP — gọi mp.spawn từ notebook
  - run_single_training      : Entry point single-GPU — dùng cho HPO shard runner
"""

import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)
from tqdm import tqdm

from torch.amp import GradScaler, autocast

from .dataset import get_dataloaders
from .models import build_model, predict_labels
from .evaluate import compute_qwk
from .utils import save_checkpoint, setup_wandb, save_wandb_run_meta, append_pipeline_log_line


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ── Shared optimizer builder ──────────────────────────────────────────────────

def _build_optimizer(model, cfg: dict, is_ddp: bool = False) -> torch.optim.AdamW:
    """
    Tạo AdamW với differential LR (backbone < head).
    Đọc BACKBONE_LR_SCALE, WEIGHT_DECAY, ADAMW_BETA1/2, ADAMW_EPS từ cfg.

    BUG FIX (so với phiên bản cũ):
      - BACKBONE_LR_SCALE: trước đây hardcode * 0.1; giờ đọc từ cfg
      - WEIGHT_DECAY: trước đây hardcode 0.05; giờ đọc từ cfg
      - ADAMW_BETA1/2: trước đây không truyền betas vào AdamW; giờ đọc từ cfg
    """
    backbone = model.module.backbone if is_ddp else model.backbone
    model_type = cfg["MODEL_TYPE"]
    is_transformer = "swin" in model_type

    if is_transformer:
        head_params = list(backbone.norm.parameters()) + \
                      list(backbone.get_classifier().parameters())
    else:
        head_params = list(backbone.get_classifier().parameters())

    head_param_ids = {id(p) for p in head_params}
    base_params = [
        p for p in model.parameters()
        if id(p) not in head_param_ids and p.requires_grad
    ]

    # ── Đọc từ cfg (không hardcode) ──────────────────────────────────────────
    backbone_lr_scale = cfg.get("BACKBONE_LR_SCALE", 0.1)
    backbone_lr = cfg["LR"] * backbone_lr_scale

    # Fallback: Transformer dùng wd cao hơn CNN nếu cfg không set
    default_wd = 0.05 if is_transformer else 1e-5
    wd = cfg.get("WEIGHT_DECAY", default_wd)

    betas = (cfg.get("ADAMW_BETA1", 0.9), cfg.get("ADAMW_BETA2", 0.999))
    eps   = cfg.get("ADAMW_EPS", 1e-8)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": backbone_lr},
            {"params": head_params,  "lr": cfg["LR"]},
        ],
        weight_decay=wd,
        betas=betas,
        eps=eps,
    )
    return optimizer


# ── Shared scheduler builder ──────────────────────────────────────────────────

def _build_scheduler(optimizer, cfg: dict, is_transformer: bool):
    """LinearWarmup + CosineWarmRestarts cho Transformer; ReduceLROnPlateau cho CNN."""
    if is_transformer:
        warmup_epochs = cfg.get("WARMUP_EPOCHS", 2)
        T_0 = cfg.get("T_0") or max(cfg.get("EPOCHS", 20) - warmup_epochs, 1)
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=int(T_0),
                                                   T_mult=1, eta_min=1e-7)
        return SequentialLR(optimizer,
                            schedulers=[warmup_sched, cosine_sched],
                            milestones=[warmup_epochs])
    else:
        return ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg.get("LR_FACTOR", 0.5),
            patience=cfg.get("LR_PATIENCE", 3),
            min_lr=1e-7,
        )


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler, rank=0, grad_accum_steps=1) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="  Train", leave=False, disable=(rank != 0))
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            outputs = model(images).reshape(-1)
            loss = criterion(outputs, labels)
            # Chia loss để gradient tích lũy tương đương full batch
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps  # log loss gốc (chưa chia)
        preds = predict_labels(outputs.detach())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.long().cpu().tolist())

        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

    avg_loss = total_loss / len(loader)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    qwk = compute_qwk(all_labels, all_preds)
    return {"loss": avg_loss, "acc": acc, "qwk": qwk}


# ── Validation epoch ───────────────────────────────────────────────────────────

def val_epoch(model, loader, criterion, device, rank=0) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False, disable=(rank != 0))
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast("cuda"):
                outputs = model(images).reshape(-1)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.float().cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    avg_loss = total_loss / len(loader)

    import numpy as np
    from .evaluate import OptimizedRounder
    rounder = OptimizedRounder()
    best_coef = rounder.fit(np.array(all_preds), np.array(all_labels))
    final_preds = np.digitize(np.array(all_preds), best_coef).tolist()

    acc = sum(p == l for p, l in zip(final_preds, all_labels)) / len(all_labels)
    qwk = compute_qwk(all_labels, final_preds)
    return {"loss": avg_loss, "acc": acc, "qwk": qwk, "coef": best_coef}


# ── Single-GPU training (HPO shard runner) ─────────────────────────────────────

def run_single_training(cfg: dict) -> None:
    """
    Training đơn GPU — không DDP, không mp.spawn.
    Dùng cho HPO shard runner (run_optuna_shard*.py).

    BUG FIX: Hàm này không tồn tại ở phiên bản trước khiến cả 2 shard runner
    (EfficientNet và SwinV2) bị ImportError ngay lần đầu chạy.

    Đọc đúng AUGMENT_VERSION (BUG FIX: cũ dùng AUG_VERSION).
    """
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BUG FIX: cfg dùng "AUGMENT_VERSION"; cũ dùng "AUG_VERSION" → luôn fallback v1
    aug_key = cfg.get("AUGMENT_VERSION", cfg.get("AUG_VERSION", "v1"))

    train_loader, val_loader, _, _, train_sampler = get_dataloaders(
        data_dir=cfg["DATA_DIR"],
        image_size=cfg["IMAGE_SIZE"],
        batch_size=cfg["BATCH_SIZE"],
        aug_version=aug_key,
        num_workers=cfg["NUM_WORKERS"],
        train_ratio=cfg["TRAIN_RATIO"],
        val_ratio=cfg["VAL_RATIO"],
        seed=cfg["SEED"],
        rank=0,
        world_size=1,
        preprocessing_strategy=cfg.get("PREPROCESSING_STRATEGY", "none"),
    )

    model = build_model(
        cfg["MODEL_TYPE"],
        freeze_strategy=cfg.get("FREEZE_STRATEGY", "none"),
    ).to(device)

    model_type    = cfg["MODEL_TYPE"]
    is_transformer = "swin" in model_type

    optimizer  = _build_optimizer(model, cfg, is_ddp=False)
    criterion  = nn.SmoothL1Loss()
    scheduler  = _build_scheduler(optimizer, cfg, is_transformer)
    scaler     = GradScaler("cuda", enabled=cfg.get("USE_AMP", True))

    sched_name = "LinearWarmup+CosineWarmRestarts" if is_transformer else "ReduceLROnPlateau"
    print(f"[Loss]      SmoothL1Loss")
    print(f"[Scheduler] {sched_name}")
    print(
        f"[Optimizer] AdamW | backbone_lr={cfg['LR'] * cfg.get('BACKBONE_LR_SCALE', 0.1):.2e}"
        f"  head_lr={cfg['LR']:.2e}"
        f"  wd={cfg.get('WEIGHT_DECAY', 0.05 if is_transformer else 1e-5):.1e}"
        f"  betas=({cfg.get('ADAMW_BETA1', 0.9)},{cfg.get('ADAMW_BETA2', 0.999)})"
    )

    # ── W&B: khởi tạo run cho single-GPU (HPO shard) ─────────────────────────
    wandb_run = setup_wandb(cfg)

    os.makedirs(os.path.join(cfg["OUTPUT_DIR"], "checkpoints"), exist_ok=True)
    best_val_qwk = -1.0
    history = {
        "train_loss": [], "train_acc": [], "train_qwk": [],
        "val_loss":   [], "val_acc":   [], "val_qwk":   [],
    }
    grad_accum_steps = cfg.get("GRAD_ACCUM_STEPS", 1)

    try:
        for epoch in range(1, cfg["EPOCHS"] + 1):
            # BUG FIX: set_epoch() chỉ có trên DistributedSampler, không có trên WeightedRandomSampler
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            t_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                    grad_accum_steps=grad_accum_steps)
            v_metrics = val_epoch(model, val_loader, criterion, device)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(v_metrics["qwk"])
            else:
                scheduler.step()

            epoch_time = time.time() - epoch_start

            for k in ["loss", "acc", "qwk"]:
                history[f"train_{k}"].append(t_metrics[k])
                history[f"val_{k}"].append(v_metrics[k])

            cf_str = ", ".join([f"{c:.2f}" for c in v_metrics.get("coef", [])])
            print(
                f"Epoch [{epoch:02d}/{cfg['EPOCHS']}] | {epoch_time:.1f}s"
                f" | Train loss: {t_metrics['loss']:.4f}  QWK: {t_metrics['qwk']:.4f}"
                f" | Val   loss: {v_metrics['loss']:.4f}  QWK: {v_metrics['qwk']:.4f}"
                f" | coef: [{cf_str}]"
            )

            # ── W&B: log metrics mỗi epoch ────────────────────────────────────────
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "epoch":           epoch,
                    "train/loss":      t_metrics["loss"],
                    "train/acc":       t_metrics["acc"],
                    "train/qwk":       t_metrics["qwk"],
                    "val/loss":        v_metrics["loss"],
                    "val/acc":         v_metrics["acc"],
                    "val/qwk":         v_metrics["qwk"],
                    "epoch_time_s":    epoch_time,
                    "lr/head":         optimizer.param_groups[1]["lr"],
                    "lr/backbone":     optimizer.param_groups[0]["lr"],
                })

            if v_metrics["qwk"] > best_val_qwk:
                best_val_qwk = v_metrics["qwk"]
                ckpt_path = os.path.join(
                    cfg["OUTPUT_DIR"], "checkpoints",
                    f"best_model_{cfg['MODEL_TYPE']}.pth",
                )
                save_checkpoint(model, optimizer, epoch, v_metrics, ckpt_path)
                print(f"  ✓ Best model saved (val QWK = {best_val_qwk:.4f})")

                # ── W&B: đánh dấu epoch tốt nhất ─────────────────────────────────
                if wandb_run is not None:
                    import wandb
                    wandb.log({"best/val_qwk": best_val_qwk, "best/epoch": epoch})

        # ── W&B: kết thúc run ────────────────────────────────────────────────────
        if wandb_run is not None:
            import wandb
            wandb.log({
                "pipeline/stage":    "training_complete",
                "train/best_val_qwk": float(best_val_qwk),
            })
            save_wandb_run_meta(cfg["OUTPUT_DIR"], wandb.run)
            wandb.finish()

        import json
        history_path = os.path.join(cfg["OUTPUT_DIR"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining hoàn tất. Best val QWK = {best_val_qwk:.4f}")

    finally:
        # ── Giải phóng GPU memory bất kể thành công hay OOM ──────────────────
        import gc
        try:
            del model, optimizer, scheduler, scaler, criterion
            del train_loader, val_loader
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ── Worker (1 process per GPU) — DDP ─────────────────────────────────────────

def train_worker(rank: int, world_size: int, cfg: dict) -> None:
    warnings.filterwarnings("ignore")
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        append_pipeline_log_line(
            cfg["OUTPUT_DIR"],
            "\n--- DDP training (rank 0) ---\n",
            rank,
        )

    # BUG FIX: đọc AUGMENT_VERSION thay vì AUG_VERSION
    aug_key = cfg.get("AUGMENT_VERSION", cfg.get("AUG_VERSION", "v1"))

    train_loader, val_loader, _, _, train_sampler = get_dataloaders(
        data_dir=cfg["DATA_DIR"],
        image_size=cfg["IMAGE_SIZE"],
        batch_size=cfg["BATCH_SIZE"],
        aug_version=aug_key,
        num_workers=cfg["NUM_WORKERS"],
        train_ratio=cfg["TRAIN_RATIO"],
        val_ratio=cfg["VAL_RATIO"],
        seed=cfg["SEED"],
        rank=rank,
        world_size=world_size,
        preprocessing_strategy=cfg.get("PREPROCESSING_STRATEGY", "none"),
    )

    model = build_model(
        cfg["MODEL_TYPE"],
        freeze_strategy=cfg.get("FREEZE_STRATEGY", "none"),
    ).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    model_type    = cfg["MODEL_TYPE"]
    is_transformer = "swin" in model_type

    # BUG FIX: dùng _build_optimizer thay vì hardcode (BACKBONE_LR_SCALE + WD + betas)
    optimizer = _build_optimizer(model, cfg, is_ddp=True)
    criterion = nn.SmoothL1Loss()
    scheduler = _build_scheduler(optimizer, cfg, is_transformer)
    scaler    = GradScaler("cuda", enabled=cfg.get("USE_AMP", True))

    if rank == 0:
        sched_name = "LinearWarmup+CosineWarmRestarts" if is_transformer else "ReduceLROnPlateau"
        print(f"[Loss]      SmoothL1Loss")
        print(f"[Scheduler] {sched_name}")
        print(
            f"[Optimizer] AdamW | backbone_lr={cfg['LR'] * cfg.get('BACKBONE_LR_SCALE', 0.1):.2e}"
            f"  head_lr={cfg['LR']:.2e}"
            f"  wd={cfg.get('WEIGHT_DECAY', 0.05 if is_transformer else 1e-5):.1e}"
            f"  betas=({cfg.get('ADAMW_BETA1', 0.9)},{cfg.get('ADAMW_BETA2', 0.999)})"
        )

    wandb_run = None
    if rank == 0:
        wandb_run = setup_wandb(cfg)

    os.makedirs(os.path.join(cfg["OUTPUT_DIR"], "checkpoints"), exist_ok=True)
    best_val_qwk = -1.0
    history = {
        "train_loss": [], "train_acc": [], "train_qwk": [],
        "val_loss":   [], "val_acc":   [], "val_qwk":   [],
    }

    for epoch in range(1, cfg["EPOCHS"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        t_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, rank)
        v_metrics = val_epoch(model, val_loader, criterion, device, rank)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(v_metrics["qwk"])
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        if rank == 0:
            for k in ["loss", "acc", "qwk"]:
                history[f"train_{k}"].append(t_metrics[k])
                history[f"val_{k}"].append(v_metrics[k])

            cf_str = ", ".join([f"{c:.2f}" for c in v_metrics.get("coef", [])])
            _line = (
                f"Epoch [{epoch:02d}/{cfg['EPOCHS']}] | {epoch_time:.1f}s"
                f" | Train loss: {t_metrics['loss']:.4f}  QWK: {t_metrics['qwk']:.4f}"
                f" | Val   loss: {v_metrics['loss']:.4f}  QWK: {v_metrics['qwk']:.4f}"
                f" | coef: [{cf_str}]"
            )
            print(_line)
            append_pipeline_log_line(cfg["OUTPUT_DIR"], _line, rank)

            if wandb_run is not None:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": t_metrics["loss"], "train/acc": t_metrics["acc"],
                    "train/qwk":  t_metrics["qwk"],
                    "val/loss":   v_metrics["loss"], "val/acc": v_metrics["acc"],
                    "val/qwk":    v_metrics["qwk"],
                    "epoch_time_s": epoch_time,
                    "lr/head":     optimizer.param_groups[1]["lr"],
                    "lr/backbone": optimizer.param_groups[0]["lr"],
                })

            if v_metrics["qwk"] > best_val_qwk:
                best_val_qwk = v_metrics["qwk"]
                ckpt_path = os.path.join(
                    cfg["OUTPUT_DIR"], "checkpoints",
                    f"best_model_{cfg['MODEL_TYPE']}.pth",
                )
                save_checkpoint(model.module, optimizer, epoch, v_metrics, ckpt_path)
                _bm = f"  ✓ Best model saved (val QWK = {best_val_qwk:.4f})"
                print(_bm)
                append_pipeline_log_line(cfg["OUTPUT_DIR"], _bm, rank)

    if rank == 0:
        if wandb_run is not None:
            import wandb
            wandb.log({"pipeline/stage": "training_complete",
                       "train/best_val_qwk": float(best_val_qwk)})
            save_wandb_run_meta(cfg["OUTPUT_DIR"], wandb.run)
            wandb.finish()
        import json
        history_path = os.path.join(cfg["OUTPUT_DIR"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        _done = f"\nTraining hoàn tất. Best val QWK = {best_val_qwk:.4f}"
        print(_done)
        append_pipeline_log_line(cfg["OUTPUT_DIR"], _done.strip(), rank)

    cleanup_ddp()


# ── Entry point DDP ───────────────────────────────────────────────────────────

def run_training(cfg: dict) -> None:
    """Khởi động DDP training bằng mp.spawn. Gọi từ notebook."""
    warnings.filterwarnings("ignore")
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("Không tìm thấy GPU.")

    repo_root = cfg.get("REPO_ROOT", "")
    if repo_root:
        sep = ":" if os.name != "nt" else ";"
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = repo_root + (sep + existing if existing else "")

    print(f"Bắt đầu DDP training trên {world_size} GPU(s)...")
    mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size, join=True)