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

# Dùng torch.amp thay vì torch.cuda.amp (tránh DeprecationWarning từ PyTorch 2.4+)
from torch.amp import GradScaler, autocast

from .dataset import get_dataloaders
from .models import build_model, predict_labels
from .evaluate import compute_qwk
from .utils import save_checkpoint, setup_wandb, save_wandb_run_meta, append_pipeline_log_line


# ── Custom Loss ───────────────────────────────────────────────────────────────

class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss có trọng số theo class — giải quyết mất cân bằng nhãn
    trong Ordinal Regression.

    Cơ chế:
      - Loss cơ bản (SmoothL1, không reduce) được nhân với trọng số tương
        ứng nhãn thật của từng sample.
      - Nhãn hiếm (e.g., class 3, 4) có weight cao → model bị phạt nặng hơn
        khi đoán sai, kéo per-class recall và QWK lên.

    Args:
        class_weights (list[float]): 5 trọng số cho class 0 → 4.
            Nên tính bằng: n_samples / (n_classes × n_samples_in_class)
        beta (float): Ngưỡng chuyển giữa L1 và L2 trong Smooth L1. Default = 1.0
    """

    def __init__(self, class_weights: list, beta: float = 1.0):
        super().__init__()
        # register_buffer → tensor tự động đẩy lên GPU cùng model.to(device)
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )
        self.beta = beta

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth L1 reduction='none' → shape [B]
        loss = F.smooth_l1_loss(preds, targets, reduction="none", beta=self.beta)

        # targets là float, cần đổi sang long để làm index vào class_weights
        target_indices = torch.round(targets).long().clamp(0, 4)
        weights = self.class_weights[target_indices]  # shape [B]

        return (loss * weights).mean()


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

        with autocast("cuda"):
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

            with autocast("cuda"):
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
    # Tắt toàn bộ warning trong child processes (FutureWarning, UserWarning, v.v.)
    warnings.filterwarnings("ignore")

    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        append_pipeline_log_line(
            cfg["OUTPUT_DIR"],
            "\n--- DDP training (rank 0) — log từ worker, không qua tee notebook ---\n",
            rank,
        )

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
    freeze_strategy = cfg.get("FREEZE_STRATEGY", "none")
    model = build_model(
        cfg["MODEL_TYPE"], freeze_strategy=freeze_strategy
    ).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    model_type: str = cfg["MODEL_TYPE"]
    is_transformer = "swin" in model_type  # phân biệt CNN vs Transformer

    # ── Optimizer & Differential Learning Rate ────────────────────────────────
    # Tách head và backbone để đặt LR khác nhau (differential LR)
    head_params   = list(model.module.backbone.get_classifier().parameters())
    head_param_ids = {id(p) for p in head_params}
    base_params   = [
        p for p in model.module.parameters()
        if id(p) not in head_param_ids and p.requires_grad
    ]

    # Transformer cần weight_decay lớn hơn CNN để chống overfit
    wd = 0.05 if is_transformer else 1e-5

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg["LR"] * 0.1},  # backbone: LR nhỏ hơn 10×
            {"params": head_params,  "lr": cfg["LR"]},         # head: LR đầy đủ
        ],
        weight_decay=wd,
    )

    # ── Loss (Criterion) ──────────────────────────────────────────────────────
    # Dùng WeightedSmoothL1Loss nếu class_weights được truyền vào, ngược lại
    # fallback về SmoothL1Loss thuần tuý.
    raw_cw = cfg.get("CLASS_WEIGHTS", None)
    if raw_cw is not None:
        criterion = WeightedSmoothL1Loss(class_weights=raw_cw).to(device)
        if rank == 0:
            print(f"[Loss] WeightedSmoothL1Loss | class_weights = {raw_cw}")
    else:
        criterion = nn.SmoothL1Loss()
        if rank == 0:
            print("[Loss] SmoothL1Loss (không có class_weights)")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # EfficientNet (CNN): ReduceLROnPlateau — an toàn, theo dõi val_qwk trực tiếp.
    # SwinV2 (Transformer): Linear Warmup (2 epoch) → CosineAnnealingWarmRestarts.
    if is_transformer:
        warmup_epochs = cfg.get("WARMUP_EPOCHS", 2)
        T_0 = cfg.get("T_0", max(cfg["EPOCHS"] - warmup_epochs, 1))
        # Warmup: LR tăng tuyến tính từ LR×0.01 → LR trong warmup_epochs epoch
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Cosine: sau warmup, decay hình cosine và restart
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=1e-7
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        scheduler_type = "LinearWarmup+CosineAnnealingWarmRestarts"
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",          # tối đa hoá val_qwk
            factor=cfg.get("LR_FACTOR", 0.5),
            patience=cfg.get("LR_PATIENCE", 3),
            min_lr=1e-7,
        )
        scheduler_type = "ReduceLROnPlateau"

    if rank == 0:
        print(f"[Scheduler] {scheduler_type}")
        print(f"[Optimizer] AdamW | backbone_lr={cfg['LR'] * 0.1:.2e}  head_lr={cfg['LR']:.2e}  wd={wd}")

    scaler = GradScaler("cuda", enabled=cfg.get("USE_AMP", True))

    # ── W&B (chỉ rank 0) ──────────────────────────────────────────────────────
    wandb_run = None
    if rank == 0:
        wandb_run = setup_wandb(cfg)
        if wandb_run is not None:
            append_pipeline_log_line(
                cfg["OUTPUT_DIR"],
                f"W&B run (training): {getattr(wandb_run, 'url', wandb_run)}",
                rank,
            )

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

        # ── Step Scheduler ────────────────────────────────────────────────────
        # ReduceLROnPlateau cần metric; LinearWarmup/Cosine chỉ cần .step()
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(v_metrics["qwk"])
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        if rank == 0:
            # Ghi history
            for k in ["loss", "acc", "qwk"]:
                history[f"train_{k}"].append(t_metrics[k])
                history[f"val_{k}"].append(v_metrics[k])

            # Log ra console + file pipeline (DDP không dùng tee của notebook)
            _line = (
                f"Epoch [{epoch:02d}/{cfg['EPOCHS']}] "
                f"| Time: {epoch_time:.1f}s "
                f"| Train loss: {t_metrics['loss']:.4f}  acc: {t_metrics['acc']:.4f}  QWK: {t_metrics['qwk']:.4f} "
                f"| Val   loss: {v_metrics['loss']:.4f}  acc: {v_metrics['acc']:.4f}  QWK: {v_metrics['qwk']:.4f}"
            )
            print(_line)
            append_pipeline_log_line(cfg["OUTPUT_DIR"], _line, rank)

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
                    "lr/head":    optimizer.param_groups[1]["lr"],
                    "lr/backbone": optimizer.param_groups[0]["lr"],
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
                _bm = f"  ✓ Best model saved  (val QWK = {best_val_qwk:.4f})"
                print(_bm)
                append_pipeline_log_line(cfg["OUTPUT_DIR"], _bm, rank)

    if rank == 0:
        if wandb_run is not None:
            import wandb
            # Giai đoạn training kết thúc — log + lưu id run để notebook resume và log tiếp eval/submit/figures
            wandb.log({
                "pipeline/stage": "training_complete",
                "train/best_val_qwk": float(best_val_qwk),
            })
            save_wandb_run_meta(cfg["OUTPUT_DIR"], wandb.run)
            wandb.finish()
        # Lưu history để notebook đọc lại
        import json
        history_path = os.path.join(cfg["OUTPUT_DIR"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        _done = f"\nTraining hoàn tất. Best val QWK = {best_val_qwk:.4f}"
        print(_done)
        append_pipeline_log_line(cfg["OUTPUT_DIR"], _done.strip(), rank)

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
    # Tắt warning trong main process (FutureWarning từ mp.spawn, v.v.)
    warnings.filterwarnings("ignore")

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
