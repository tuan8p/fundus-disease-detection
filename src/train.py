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

"""
train.py
--------
Pipeline huấn luyện APTOS 2019 & WeightedRandomSampler.
Hỗ trợ chạy Local Pass (1 GPU) hoặc DDP (Multi-GPU).
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
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
#Dùng để chạy grid
############
import copy
import itertools
import json
import glob
import shutil
##################
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
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False, disable=(rank != 0))

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            # Trả về [B, 5] cho bài toán Classification
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # predict_labels bây giờ dùng argmax
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
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False, disable=(rank != 0))

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images).view(-1)
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
    warnings.filterwarnings("ignore")

    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        append_pipeline_log_line(
            cfg["OUTPUT_DIR"],
            f"\n--- Bắt đầu huấn luyện trên {'DDP (' + str(world_size) + ' GPUs)' if use_ddp else 'Local (1 GPU)'} ---\n",
            rank,
        )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, _, _, train_sampler = get_dataloaders(
        data_dir=cfg["DATA_DIR"],
        image_size=cfg["IMAGE_SIZE"],
        batch_size=cfg["BATCH_SIZE"],
        aug_version=cfg.get("AUGMENT_VERSION", "v1"),
        num_workers=cfg["NUM_WORKERS"],
        train_ratio=cfg["TRAIN_RATIO"],
        val_ratio=cfg["VAL_RATIO"],
        seed=cfg["SEED"],
        rank=rank,
        world_size=world_size,
        preprocessing_strategy=cfg.get("PREPROCESSING_STRATEGY", "none"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg["MODEL_TYPE"]).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # ── Optimizer & Loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    criterion = nn.SmoothL1Loss()
    scaler    = GradScaler("cuda", enabled=cfg.get("USE_AMP", True))
    
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
        
        # BẢO VỆ SAMPLER: Chỉ gọi set_epoch nếu là DDP Sampler
        if train_sampler is not None and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        t_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, rank)
        v_metrics = val_epoch(model, val_loader, criterion, device, rank)

        epoch_time = time.time() - epoch_start

        if rank == 0:
            for k in ["loss", "acc", "qwk"]:
                history[f"train_{k}"].append(t_metrics[k])
                history[f"val_{k}"].append(v_metrics[k])

            _line = (
                f"Epoch [{epoch:02d}/{cfg['EPOCHS']}] "
                f"| Time: {epoch_time:.1f}s "
                f"| Train loss: {t_metrics['loss']:.4f}  acc: {t_metrics['acc']:.4f}  QWK: {t_metrics['qwk']:.4f} "
                f"| Val   loss: {v_metrics['loss']:.4f}  acc: {v_metrics['acc']:.4f}  QWK: {v_metrics['qwk']:.4f}"
            )
            print(_line)
            append_pipeline_log_line(cfg["OUTPUT_DIR"], _line, rank)

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

            if v_metrics["qwk"] > best_val_qwk:
                best_val_qwk = v_metrics["qwk"]
                ckpt_path = os.path.join(
                    cfg["OUTPUT_DIR"], "checkpoints",
                    f"best_model_{cfg['MODEL_TYPE']}.pth"
                )
                
                # Lưu state_dict: Xử lý an toàn cho cả model thường và model bọc DDP
                model_to_save = model.module if hasattr(model, 'module') else model
                save_checkpoint(model_to_save, optimizer, epoch, v_metrics, ckpt_path)
                
                _bm = f"  ✓ Best model saved  (val QWK = {best_val_qwk:.4f})"
                print(_bm)
                append_pipeline_log_line(cfg["OUTPUT_DIR"], _bm, rank)

    if rank == 0:
        if wandb_run is not None:
            import wandb
            wandb.log({
                "pipeline/stage": "training_complete",
                "train/best_val_qwk": float(best_val_qwk),
            })
            save_wandb_run_meta(cfg["OUTPUT_DIR"], wandb.run)
            wandb.finish()
            
        import json
        history_path = os.path.join(cfg["OUTPUT_DIR"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
            
        _done = f"\nTraining hoàn tất. Best val QWK = {best_val_qwk:.4f}"
        print(_done)
        append_pipeline_log_line(cfg["OUTPUT_DIR"], _done.strip(), rank)

    if use_ddp:
        cleanup_ddp()


# ── Entry point ───────────────────────────────────────────────────────────────


def run_single_training(cfg: dict) -> None:
        """
        Hàm huấn luyện đơn lẻ (Chính là hàm run_training cũ của bạn).
        """
        warnings.filterwarnings("ignore")
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("Không tìm thấy GPU. Hãy kiểm tra lại môi trường.")

        repo_root = cfg.get("REPO_ROOT", "")
        if repo_root:
            sep = ":" if os.name != "nt" else ";"
            existing = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = repo_root + (sep + existing if existing else "")

        if world_size > 1:
            print(f"Bắt đầu DDP training trên {world_size} GPU(s)...")
            mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size, join=True)
        else:
            print("Phát hiện 1 GPU. Bắt đầu huấn luyện Local Pass...")
            train_worker(rank=0, world_size=1, cfg=cfg)


def run_grid_search(base_cfg: dict) -> None:
        """
        Trình điều khiển chạy nhiều thí nghiệm liên tiếp.
        """
        # Ép Epochs = 5 để chống sập server Kaggle khi chạy 12 models
        grid_cfg = copy.deepcopy(base_cfg)
        grid_cfg["EPOCHS"] = min(grid_cfg.get("EPOCHS", 5), 5) 
        
        preproc_strategies = ["none", "roi", "ben", "clahe"]
        aug_versions       = ["v1", "v2", "v3"]
        experiments = list(itertools.product(preproc_strategies, aug_versions))

        main_output_dir = grid_cfg["OUTPUT_DIR"]
        best_overall_qwk = -1.0
        winning_exp_dir = ""

        print(f"🚀 KÍCH HOẠT AUTO GRID SEARCH: Chạy {len(experiments)} thí nghiệm (Epoch={grid_cfg['EPOCHS']})")

        for idx, (preproc, aug) in enumerate(experiments, start=1):
            current_cfg = copy.deepcopy(grid_cfg)
            current_cfg["PREPROCESSING_STRATEGY"] = preproc
            current_cfg["AUGMENT_VERSION"] = aug
            
            exp_name = f"exp_{idx:02d}_{preproc}_{aug}"
            exp_dir = os.path.join(main_output_dir, exp_name)
            current_cfg["OUTPUT_DIR"] = exp_dir
            current_cfg["WANDB_RUN_NAME"] = f"{grid_cfg['MODEL_TYPE']}_{exp_name}"

            print("\n" + "="*70)
            print(f"🔥 ĐANG CHẠY THÍ NGHIỆM {idx}/{len(experiments)}: [{preproc.upper()}] + [{aug.upper()}]")
            print("="*70)

            try:
                run_single_training(current_cfg)

                # Đọc file history để tìm model ngon nhất
                history_path = os.path.join(exp_dir, "history.json")
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        hist = json.load(f)
                        max_val_qwk = max(hist.get("val_qwk", [0]))
                        if max_val_qwk > best_overall_qwk:
                            best_overall_qwk = max_val_qwk
                            winning_exp_dir = exp_dir
            except Exception as e:
                print(f"❌ Lỗi văng tại {exp_name}: {e}")

        print("\n" + "🎉"*15)
        print(f"GRID SEARCH HOÀN TẤT! NHÀ VÔ ĐỊCH: {os.path.basename(winning_exp_dir)} (QWK = {best_overall_qwk:.4f})")

        # BƯỚC QUAN TRỌNG: Đẩy model vô địch ra ngoài cho các Cell 6,7,8 của Notebook dùng
        if winning_exp_dir:
            print("📥 Đang trích xuất Weights & History của nhà vô địch ra thư mục gốc...")
            os.makedirs(os.path.join(main_output_dir, "checkpoints"), exist_ok=True)
            
            # Copy Weights
            for file in glob.glob(os.path.join(winning_exp_dir, "checkpoints", "*.pth")):
                shutil.copy(file, os.path.join(main_output_dir, "checkpoints", os.path.basename(file)))
            
            # Copy History
            history_src = os.path.join(winning_exp_dir, "history.json")
            if os.path.exists(history_src):
                shutil.copy(history_src, os.path.join(main_output_dir, "history.json"))
            
            print("✅ Đã sẵn sàng cho các Cell Evaluation!")


def run_training(cfg: dict) -> None:
        """
        Router (Bộ định tuyến) thay thế cho hàm cũ.
        """
        preproc = cfg.get("PREPROCESSING_STRATEGY", "none")
        aug = cfg.get("AUGMENT_VERSION", "v1")

        # Nếu User truyền vào chữ "all", ta bẻ lái sang luồng Grid Search
        if preproc == "all" or aug == "all":
            run_grid_search(cfg)
        else:
            # Chạy bình thường
            run_single_training(cfg)