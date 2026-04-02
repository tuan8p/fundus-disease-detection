"""
evaluate.py
-----------
Tính toán toàn bộ metrics đánh giá model trên tập held-out.

Metrics được hỗ trợ:
  - Quadratic Weighted Kappa (QWK)   ← metric chính
  - Accuracy
  - Macro F1-score
  - Balanced Accuracy
  - Per-class Recall
  - Confusion Matrix (numpy array)
  - Classification Report (text)
  - Validation Loss

Hàm chính: run_evaluation() — chạy toàn bộ pipeline evaluation trên 1 DataLoader.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_qwk(y_true: list, y_pred: list) -> float:
    """
    Tính Quadratic Weighted Kappa (QWK).
    Trả về 0.0 nếu chỉ có 1 lớp duy nhất trong y_true (tránh lỗi sklearn).
    """
    if len(set(y_true)) < 2:
        return 0.0
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Tính toàn bộ metrics từ danh sách nhãn thật và nhãn dự đoán.

    Args:
        y_true (list[int]): Nhãn thật  (0–4)
        y_pred (list[int]): Nhãn dự đoán (0–4)

    Returns:
        dict với các key:
            qwk, accuracy, macro_f1, balanced_accuracy,
            per_class_recall (list), confusion_matrix (ndarray)
    """
    labels = list(range(5))  # [0, 1, 2, 3, 4]

    qwk              = compute_qwk(y_true, y_pred)
    accuracy         = accuracy_score(y_true, y_pred)
    macro_f1         = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    balanced_acc     = balanced_accuracy_score(y_true, y_pred)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels).tolist()
    cm               = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "qwk":               qwk,
        "accuracy":          accuracy,
        "macro_f1":          macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_recall":  per_class_recall,
        "confusion_matrix":  cm,
    }


def build_evaluation_metrics_text(val_loss: float, metrics: dict, report_str: str) -> str:
    """
    Ghép chuỗi text đầy đủ giống output console (để lưu file / W&B).
    """
    lines = [
        "=" * 60,
        "EVALUATION RESULTS (Internal Test Set — Held-out 10%)",
        "=" * 60,
        f"  Validation Loss      : {val_loss:.4f}",
        f"  QWK (primary)        : {metrics['qwk']:.4f}",
        f"  Accuracy             : {metrics['accuracy']:.4f}",
        f"  Macro F1             : {metrics['macro_f1']:.4f}",
        f"  Balanced Accuracy    : {metrics['balanced_accuracy']:.4f}",
        f"  Per-class Recall     : {[round(r, 4) for r in metrics['per_class_recall']]}",
        "",
        report_str.rstrip(),
        "=" * 60,
    ]
    return "\n".join(lines) + "\n"


def get_classification_report(y_true: list, y_pred: list) -> str:
    """
    Trả về classification report dạng text (sklearn).

    Args:
        y_true, y_pred: danh sách nhãn nguyên 0–4
    Returns:
        str: báo cáo chi tiết per-class precision / recall / f1
    """
    target_names = [f"Class {i}" for i in range(5)]
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
        labels=list(range(5)),
    )


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    output_dir: str,
) -> dict:
    """
    Chạy toàn bộ evaluation trên 1 DataLoader (thường là internal test set).

    Kết quả:
      - In metrics ra console
      - Lưu classification_report.txt vào output_dir
      - Trả về dict metrics + y_true + y_pred để notebook dùng tiếp (vẽ figure)

    Args:
        model      : Model đã load best checkpoint
        loader     : DataLoader của tập held-out (có nhãn)
        criterion  : Loss function (SmoothL1)
        device     : torch.device
        output_dir : Thư mục lưu báo cáo văn bản

    Returns:
        dict: metrics + "y_true" + "y_pred" + "val_loss" + "classification_report_text"
    """
    from .models import predict_labels

    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images).reshape(-1)  # [B]
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = predict_labels(outputs)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    val_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    metrics["val_loss"] = val_loss
    metrics["y_true"]   = all_labels
    metrics["y_pred"]   = all_preds

    # ── In kết quả ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Internal Test Set — Held-out 10%)")
    print("="*60)
    print(f"  Validation Loss      : {val_loss:.4f}")
    print(f"  QWK (primary)        : {metrics['qwk']:.4f}")
    print(f"  Accuracy             : {metrics['accuracy']:.4f}")
    print(f"  Macro F1             : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy    : {metrics['balanced_accuracy']:.4f}")
    print(f"  Per-class Recall     : {[round(r, 4) for r in metrics['per_class_recall']]}")
    print()
    report_str = get_classification_report(all_labels, all_preds)
    print(report_str)
    print("="*60)

    full_text = build_evaluation_metrics_text(val_loss, metrics, report_str)
    metrics["classification_report_text"] = report_str

    # ── Lưu file text: bản đầy đủ + classification_report (tương thích cũ) ───────
    os.makedirs(output_dir, exist_ok=True)
    eval_txt_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Evaluation metrics (full) đã lưu tại: {eval_txt_path}")

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Classification report (same content) : {report_path}")

    return metrics
