"""
utils.py
--------
Các tiện ích hỗ trợ pipeline:

  - save_checkpoint      : Lưu model state_dict + optimizer + metadata
  - load_checkpoint      : Load lại checkpoint, trả về model + metadata
  - generate_submission  : Inference trên test.csv → submission.csv (format Kaggle)
  - zip_outputs          : Nén toàn bộ thư mục outputs/ thành outputs.zip
  - setup_wandb          : Khởi tạo W&B run (key từ Kaggle Secrets)
"""

import os
import zipfile
import json

import torch
import torch.nn as nn
import pandas as pd
from torch.cuda.amp import autocast
from tqdm import tqdm


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """
    Lưu checkpoint gồm model weights, optimizer state và metadata epoch.

    Args:
        model     : Model (không bọc DDP — dùng model.module khi DDP)
        optimizer : Optimizer hiện tại
        epoch     : Epoch hiện tại
        metrics   : Dict metrics (val_qwk, val_loss, ...)
        path      : Đường dẫn file .pth
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
    }, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """
    Load checkpoint vào model (và optimizer nếu cần tiếp tục training).

    Args:
        model     : Model đã được khởi tạo cùng kiến trúc
        path      : Đường dẫn file .pth
        device    : Thiết bị load tensor lên
        optimizer : Nếu không None, load luôn optimizer state

    Returns:
        dict: metadata (epoch, metrics)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Checkpoint loaded: {path}  (epoch {ckpt['epoch']}, metrics: {ckpt['metrics']})")
    return {"epoch": ckpt["epoch"], "metrics": ckpt["metrics"]}


# ── Submission CSV ─────────────────────────────────────────────────────────────

def generate_submission(
    model: nn.Module,
    submit_loader,
    image_ids: list,
    output_dir: str,
    device: torch.device,
) -> str:
    """
    Chạy inference trên tập test không nhãn, lưu kết quả thành submission.csv.
    Format giống sample_submission.csv: cột id_code, cột diagnosis.

    Args:
        model        : Model đã load best checkpoint
        submit_loader: DataLoader của test.csv (không nhãn)
        image_ids    : Danh sách id_code từ test.csv (giữ đúng thứ tự)
        output_dir   : Thư mục lưu submission.csv
        device       : torch.device

    Returns:
        str: Đường dẫn file submission.csv
    """
    from .models import predict_labels

    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in tqdm(submit_loader, desc="Generating submission"):
            images = images.to(device, non_blocking=True)
            with autocast():
                outputs = model(images).squeeze(1)
            preds = predict_labels(outputs)
            all_preds.extend(preds.cpu().tolist())

    os.makedirs(output_dir, exist_ok=True)
    sub_path = os.path.join(output_dir, "submission.csv")
    df = pd.DataFrame({"id_code": image_ids, "diagnosis": all_preds})
    df.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}  ({len(df)} rows)")
    return sub_path


# ── Zip outputs ───────────────────────────────────────────────────────────────

def zip_outputs(output_dir: str) -> str:
    """
    Nén toàn bộ thư mục output_dir thành outputs.zip ở cùng cấp.

    Args:
        output_dir : Thư mục outputs/ cần nén

    Returns:
        str: Đường dẫn file .zip
    """
    zip_path = output_dir.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for fname in files:
                file_path = os.path.join(root, fname)
                arcname   = os.path.relpath(file_path, start=os.path.dirname(output_dir))
                zf.write(file_path, arcname)
    print(f"Outputs zipped: {zip_path}")
    return zip_path


# ── W&B setup ─────────────────────────────────────────────────────────────────

def setup_wandb(cfg: dict):
    """
    Khởi tạo W&B run. API key lấy từ biến môi trường WANDB_API_KEY
    (nên được set từ Kaggle Secrets trước khi gọi hàm này).

    Args:
        cfg (dict): Thông số config từ Cell 2 notebook

    Returns:
        wandb.Run hoặc None nếu wandb không khả dụng / key trống
    """
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            print("WANDB_API_KEY trống — bỏ qua W&B logging.")
            return None

        wandb.login(key=api_key, relogin=True)
        run = wandb.init(
            project=cfg.get("WANDB_PROJECT", "fundus-baseline"),
            name=cfg.get("WANDB_RUN_NAME", cfg.get("MODEL_TYPE", "run")),
            config={
                k: v for k, v in cfg.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            },
            reinit=True,
        )
        print(f"W&B run initialized: {run.url}")
        return run
    except Exception as e:
        print(f"W&B setup thất bại: {e} — tiếp tục training không có W&B.")
        return None


# ── Load history JSON ─────────────────────────────────────────────────────────

def load_history(output_dir: str) -> dict:
    """
    Đọc history.json được lưu bởi train_worker sau khi training.

    Args:
        output_dir : Thư mục outputs/

    Returns:
        dict với keys: train_loss, train_acc, train_qwk, val_loss, val_acc, val_qwk
    """
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "r") as f:
        history = json.load(f)
    return history
