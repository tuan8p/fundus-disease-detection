"""
train.py
--------
Pipeline huấn luyện với DistributedDataParallel (DDP) trên 2× T4 GPU.

Các thành phần chính:
  - setup_ddp / cleanup_ddp  : Khởi tạo và dọn dẹp process group
  - train_epoch              : 1 epoch training với AMP + tqdm
  - val_epoch                : 1 epoch validation với AMP + tqdm
  - train_worker             : Hàm worker cho mỗi rank (spawn target)
  - run_training             : Entry point — gọi mp.spawn từ notebook
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .dataset import get_dataloaders
from .models import build_model, predict_labels
from .evaluate import compute_qwk
from .utils import save_checkpoint, setup_wandb


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp(rank: int, world_size: int) -> None:
    """Khởi tạo process group NCCL cho DDP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Hủy process group sau khi training xong."""
    dist.destroy_process_group()


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    rank: int,
) -> dict:
    """
    Chạy 1 epoch training với Mixed Precision (AMP).

    Returns:
        dict: {"loss": float, "acc": float, "qwk": float}
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # Chỉ hiển thị tqdm ở rank 0 tránh log trùng lặp
    pbar = tqdm(loader, desc="  Train", leave=False, disable=(rank != 0))

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images).reshape(-1)  # [B] — an toàn cho cả [B,1] và [B]
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = predict_labels(outputs.detach())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.long().cpu().tolist())

        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    qwk = compute_qwk(all_labels, all_preds)

    return {"loss": avg_loss, "acc": acc, "qwk": qwk}


# ── Validation epoch ───────────────────────────────────────────────────────────

def val_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
) -> dict:
    """
    Chạy 1 epoch validation với Mixed Precision (AMP), không update gradient.

    Returns:
        dict: {"loss": float, "acc": float, "qwk": float}
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False, disable=(rank != 0))

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images).reshape(-1)  # [B]
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = predict_labels(outputs)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    qwk = compute_qwk(all_labels, all_preds)

    return {"loss": avg_loss, "acc": acc, "qwk": qwk}


# ── Worker (1 process per GPU) ────────────────────────────────────────────────

def train_worker(rank: int, world_size: int, cfg: dict) -> None:
    """
    Hàm chạy trong mỗi tiến trình DDP (target của mp.spawn).

    Args:
        rank       : GPU index (0 hoặc 1)
        world_size : Tổng số GPU (2)
        cfg        : Dict thông số từ Cell 2 của notebook
    """
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, _, _, train_sampler = get_dataloaders(
        data_dir=cfg["DATA_DIR"],
        image_size=cfg["IMAGE_SIZE"],
        batch_size=cfg["BATCH_SIZE"],
        num_workers=cfg["NUM_WORKERS"],
        train_ratio=cfg["TRAIN_RATIO"],
        val_ratio=cfg["VAL_RATIO"],
        seed=cfg["SEED"],
        rank=rank,
        world_size=world_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg["MODEL_TYPE"]).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # ── Optimizer & Loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    criterion = nn.SmoothL1Loss()
    scaler    = GradScaler(enabled=cfg.get("USE_AMP", True))

    # ── W&B (chỉ rank 0) ──────────────────────────────────────────────────────
    wandb_run = None
    if rank == 0:
        wandb_run = setup_wandb(cfg)

    os.makedirs(os.path.join(cfg["OUTPUT_DIR"], "checkpoints"), exist_ok=True)
    best_val_qwk = -1.0
    history = {"train_loss": [], "train_acc": [], "train_qwk": [],
               "val_loss":   [], "val_acc":   [], "val_qwk":   []}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg["EPOCHS"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        t_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, rank)
        v_metrics = val_epoch(model, val_loader, criterion, device, rank)

        epoch_time = time.time() - epoch_start

        if rank == 0:
            # Ghi history
            for k in ["loss", "acc", "qwk"]:
                history[f"train_{k}"].append(t_metrics[k])
                history[f"val_{k}"].append(v_metrics[k])

            # Log ra console
            print(
                f"Epoch [{epoch:02d}/{cfg['EPOCHS']}] "
                f"| Time: {epoch_time:.1f}s "
                f"| Train loss: {t_metrics['loss']:.4f}  acc: {t_metrics['acc']:.4f}  QWK: {t_metrics['qwk']:.4f} "
                f"| Val   loss: {v_metrics['loss']:.4f}  acc: {v_metrics['acc']:.4f}  QWK: {v_metrics['qwk']:.4f}"
            )

            # W&B log
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": t_metrics["loss"],
                    "train/acc":  t_metrics["acc"],
                    "train/qwk":  t_metrics["qwk"],
                    "val/loss":   v_metrics["loss"],
                    "val/acc":    v_metrics["acc"],
                    "val/qwk":    v_metrics["qwk"],
                    "epoch_time_s": epoch_time,
                })

            # Lưu best model theo val QWK
            if v_metrics["qwk"] > best_val_qwk:
                best_val_qwk = v_metrics["qwk"]
                ckpt_path = os.path.join(
                    cfg["OUTPUT_DIR"], "checkpoints",
                    f"best_model_{cfg['MODEL_TYPE']}.pth"
                )
                # Lưu state_dict của module bên trong DDP wrapper
                save_checkpoint(model.module, optimizer, epoch, v_metrics, ckpt_path)
                print(f"  ✓ Best model saved  (val QWK = {best_val_qwk:.4f})")

    if rank == 0:
        if wandb_run is not None:
            import wandb
            wandb.finish()
        # Lưu history để notebook đọc lại
        import json
        history_path = os.path.join(cfg["OUTPUT_DIR"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining hoàn tất. Best val QWK = {best_val_qwk:.4f}")

    cleanup_ddp()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_training(cfg: dict) -> None:
    """
    Khởi động DDP training bằng mp.spawn.
    Gọi hàm này từ notebook (Cell 5).

    Args:
        cfg (dict): Toàn bộ thông số từ Cell 2 của notebook.
                    Phải có key "REPO_ROOT" để child processes tìm được src/.
    """
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("Không tìm thấy GPU. Hãy kiểm tra lại môi trường.")

    # mp.spawn tạo process mới với Python path trắng (không kế thừa sys.path
    # từ notebook). Fix: set PYTHONPATH env var — được kế thừa bởi child processes.
    repo_root = cfg.get("REPO_ROOT", "")
    if repo_root:
        sep = ":" if os.name != "nt" else ";"
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = repo_root + (sep + existing if existing else "")

    print(f"Bắt đầu DDP training trên {world_size} GPU(s)...")
    mp.spawn(
        train_worker,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True,
    )
