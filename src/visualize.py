"""
visualize.py
------------
Vẽ và lưu các figure đánh giá model.

Các figure được tạo:
  - plot_confusion_matrix    : Heatmap confusion matrix (seaborn)
  - plot_training_curves     : Loss + QWK theo epoch (train vs val)
  - plot_per_class_recall    : Bar chart recall từng lớp

Tất cả figure được lưu vào output_dir/figures/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


CLASS_NAMES = ["0-Normal", "1-Mild", "2-Moderate", "3-Severe", "4-Prolif."]


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: str,
    model_type: str = "",
    normalize: bool = True,
) -> str:
    """
    Vẽ heatmap confusion matrix và lưu file PNG.

    Args:
        cm         : Confusion matrix (5×5 numpy array)
        output_dir : Thư mục gốc outputs/
        model_type : Tên model (thêm vào tiêu đề)
        normalize  : Có normalize theo hàng (recall) không

    Returns:
        str: Đường dẫn file PNG đã lưu
    """
    fig_dir = os.path.join(output_dir, "figures")
    _ensure_dir(fig_dir)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1          # tránh chia 0
        cm_plot = cm.astype(float) / row_sum
        fmt = ".2f"
        title = f"Normalized Confusion Matrix\n{model_type}"
    else:
        cm_plot = cm
        fmt = "d"
        title = f"Confusion Matrix\n{model_type}"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt, cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")
    return save_path


# ── Training Curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    output_dir: str,
    model_type: str = "",
) -> str:
    """
    Vẽ 2 subplot: Loss theo epoch và QWK theo epoch (train vs val).

    Args:
        history    : Dict với keys train_loss, val_loss, train_qwk, val_qwk
        output_dir : Thư mục gốc outputs/
        model_type : Tên model (thêm vào tiêu đề)

    Returns:
        str: Đường dẫn file PNG đã lưu
    """
    fig_dir = os.path.join(output_dir, "figures")
    _ensure_dir(fig_dir)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves — {model_type}", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2563EB")
    ax.plot(epochs, history["val_loss"],   "s--", label="Val Loss",  color="#DC2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── QWK ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_qwk"], "o-", label="Train QWK", color="#059669")
    ax.plot(epochs, history["val_qwk"],   "s--", label="Val QWK",  color="#D97706")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quadratic Weighted Kappa")
    ax.set_title("QWK (Quadratic Weighted Kappa)")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved: {save_path}")
    return save_path


# ── Per-class Recall ──────────────────────────────────────────────────────────

def plot_per_class_recall(
    per_class_recall: list,
    output_dir: str,
    model_type: str = "",
) -> str:
    """
    Vẽ bar chart recall từng lớp 0–4.

    Args:
        per_class_recall : List 5 giá trị recall [class0, class1, ..., class4]
        output_dir       : Thư mục gốc outputs/
        model_type       : Tên model (thêm vào tiêu đề)

    Returns:
        str: Đường dẫn file PNG đã lưu
    """
    fig_dir = os.path.join(output_dir, "figures")
    _ensure_dir(fig_dir)

    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_NAMES, per_class_recall, color=colors, edgecolor="white", linewidth=0.8)

    # Ghi giá trị lên mỗi cột
    for bar, val in zip(bars, per_class_recall):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title(f"Per-class Recall — {model_type}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=np.mean(per_class_recall), color="gray", linestyle="--", alpha=0.7,
               label=f"Mean recall = {np.mean(per_class_recall):.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "per_class_recall.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-class recall saved: {save_path}")
    return save_path


# ── Convenience: vẽ tất cả figure cùng lúc ───────────────────────────────────

def plot_all(metrics: dict, history: dict, output_dir: str, model_type: str = "") -> list:
    """
    Gọi tất cả 3 hàm vẽ figure, trả về danh sách đường dẫn file PNG.

    Args:
        metrics    : Dict trả về từ evaluate.run_evaluation()
        history    : Dict history từ training (đọc từ history.json)
        output_dir : Thư mục gốc outputs/
        model_type : Tên model

    Returns:
        list[str]: Danh sách đường dẫn file đã lưu
    """
    paths = []
    paths.append(plot_confusion_matrix(metrics["confusion_matrix"], output_dir, model_type))
    paths.append(plot_training_curves(history, output_dir, model_type))
    paths.append(plot_per_class_recall(metrics["per_class_recall"], output_dir, model_type))
    return paths
