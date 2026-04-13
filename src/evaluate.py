import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
import scipy.optimize as opt
from functools import partial

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


# ── Threshold Optimization ──────────────────────────────────────────────────

class OptimizedRounder:
    """Tối ưu hóa ngưỡng Threshold để cực đại hóa QWK."""
    def _kappa_loss(self, coef, X, y):
        coef_sorted = np.sort(coef)          # ← thêm dòng này
        preds = np.digitize(X, coef_sorted)
        return -compute_qwk(y, preds)

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        res = opt.minimize(loss_partial, initial_coef, method='nelder-mead')
        return sorted(res.x.tolist())


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
    labels = list(range(5))  

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
    coef: list = None,
) -> dict:
    from .models import predict_labels

    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images).reshape(-1)  
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = predict_labels(outputs, coef=coef)
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
