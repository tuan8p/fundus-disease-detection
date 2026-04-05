"""
preprocessing.py
------------------
Tiền xử lý ảnh fundus (APTOS) — deterministic, áp trước Resize + ImageNet norm trong dataset.

4 strategy thực nghiệm:
  - roi              : chỉ ROI (bỏ viền đen)
  - roi_ben          : ROI + Ben Graham (scale theo bán kính ~300px)
  - roi_imgtype      : ROI + chuẩn hóa theo "loại ảnh" (độ sáng)
  - roi_ben_imgtype  : ROI + Ben + image type

Thêm "none" để tắt hoàn toàn (giữ pipeline baseline cũ: chỉ resize + norm).

Không augmentation — chỉ preprocessing.
"""

from __future__ import annotations

from typing import Callable, Final

import cv2
import numpy as np
from PIL import Image

STRATEGY_NONE: Final[str] = "none"
STRATEGY_ROI: Final[str] = "roi"
STRATEGY_ROI_BEN: Final[str] = "roi_ben"
STRATEGY_ROI_IMGT: Final[str] = "roi_imgtype"
STRATEGY_ROI_BEN_IMGT: Final[str] = "roi_ben_imgtype"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_NONE,
    STRATEGY_ROI,
    STRATEGY_ROI_BEN,
    STRATEGY_ROI_IMGT,
    STRATEGY_ROI_BEN_IMGT,
)

_DEFAULT_TOL: Final[int] = 7
_BEN_TARGET_RADIUS: Final[int] = 300
_IMGT_BRIGHT_THRESH: Final[int] = 15


def extract_roi(image: Image.Image, tol: int = _DEFAULT_TOL, pad_ratio: float = 0.05) -> Image.Image:
    """
    Cắt vùng fundus, bỏ viền đen; pad nhẹ rồi căn vuông (letterbox đen).
    Nếu không tìm được mask → trả về ảnh gốc.
    """
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return image

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    pad_y = max(1, int(h * pad_ratio))
    pad_x = max(1, int(w * pad_ratio))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(arr.shape[0], y1 + pad_y)
    x1 = min(arr.shape[1], x1 + pad_x)

    cropped = arr[y0:y1, x0:x1].copy()
    ch, cw = cropped.shape[0], cropped.shape[1]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off : y_off + ch, x_off : x_off + cw] = cropped
    return Image.fromarray(square)


def ben_graham_scale(rgb: np.ndarray, target_radius: int = _BEN_TARGET_RADIUS) -> np.ndarray:
    """
    Sau ROI, scale ảnh sao cho bán kính ước lượng (nửa cạnh ngắn) ~ target_radius.
    """
    h, w = rgb.shape[:2]
    r_est = max(min(h, w) // 2, 1)
    scale = target_radius / float(r_est)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_type_normalize(rgb: np.ndarray, bright_thresh: int = _IMGT_BRIGHT_THRESH) -> np.ndarray:
    """
    Phân nhóm theo tỷ lệ pixel gray > bright_thresh; chỉnh độ sáng nhẹ (linear).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    frac = float(np.mean(gray > bright_thresh))
    x = rgb.astype(np.float32)
    if frac < 0.12:
        x = np.clip(x * 1.10, 0, 255)
    elif frac > 0.38:
        x = np.clip(x * 0.95, 0, 255)
    return x.astype(np.uint8)


def preprocess_roi_only(image: Image.Image) -> Image.Image:
    """Strategy 4: chỉ ROI."""
    return extract_roi(image)


def preprocess_roi_ben(image: Image.Image) -> Image.Image:
    """Strategy 2: ROI + Ben scale."""
    pil = extract_roi(image)
    arr = np.array(pil)
    out = ben_graham_scale(arr)
    return Image.fromarray(out)


def preprocess_roi_imgtype(image: Image.Image) -> Image.Image:
    """Strategy 3: ROI + image type norm."""
    pil = extract_roi(image)
    arr = np.array(pil)
    out = image_type_normalize(arr)
    return Image.fromarray(out)


def preprocess_roi_ben_imgtype(image: Image.Image) -> Image.Image:
    """Strategy 1: ROI + Ben + image type."""
    pil = extract_roi(image)
    arr = np.array(pil)
    arr = ben_graham_scale(arr)
    arr = image_type_normalize(arr)
    return Image.fromarray(arr)


def preprocess_none(image: Image.Image) -> Image.Image:
    """Không đổi (baseline trước resize)."""
    return image


_PREPROCESS_FUNCS: dict[str, Callable[[Image.Image], Image.Image]] = {
    STRATEGY_NONE: preprocess_none,
    STRATEGY_ROI: preprocess_roi_only,
    STRATEGY_ROI_BEN: preprocess_roi_ben,
    STRATEGY_ROI_IMGT: preprocess_roi_imgtype,
    STRATEGY_ROI_BEN_IMGT: preprocess_roi_ben_imgtype,
}


def list_strategies() -> list[str]:
    return list(ALL_STRATEGIES)


def validate_strategy(name: str) -> str:
    if name not in ALL_STRATEGIES:
        raise ValueError(
            f"PREPROCESSING_STRATEGY='{name}' không hợp lệ. "
            f"Chọn một trong: {ALL_STRATEGIES}"
        )
    return name


def get_preprocess_fn(strategy: str) -> Callable[[Image.Image], Image.Image]:
    validate_strategy(strategy)
    return _PREPROCESS_FUNCS[strategy]


def apply_preprocessing(image: Image.Image, strategy: str) -> Image.Image:
    fn = get_preprocess_fn(strategy)
    return fn(image)
