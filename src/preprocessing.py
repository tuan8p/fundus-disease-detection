"""
preprocessing.py
------------------
Tiền xử lý ảnh fundus (APTOS) — deterministic, áp trước Resize + ImageNet norm trong dataset.

Tham số căn cứ quy trình có hệ thống (grid / tài liệu APTOS):
  - Ben Graham: unsharp mask 4*img - 4*GaussianBlur(sigmaX) + 128 (sigmaX mặc định 10)
  - CLAHE: kênh L của LAB, clipLimit / tileGridSize có thể cấu hình

Các mode:
  - none          : không tiền xử lý thêm (baseline: chỉ resize + norm trong transforms)
  - roi           : chỉ ROI (bỏ viền đen, căn vuông)
  - roi_ben       : ROI + Ben Graham
  - roi_clahe     : ROI + CLAHE (L channel)
  - roi_ben_clahe : ROI + Ben → CLAHE (thứ tự theo tài liệu: Ben trước, CLAHE sau)

Không augmentation.
"""

from __future__ import annotations

from typing import Callable, Final

import cv2
import numpy as np
from PIL import Image

STRATEGY_NONE: Final[str] = "none"
STRATEGY_ROI: Final[str] = "roi"
STRATEGY_ROI_BEN: Final[str] = "roi_ben"
STRATEGY_ROI_CLAHE: Final[str] = "roi_clahe"
STRATEGY_ROI_BEN_CLAHE: Final[str] = "roi_ben_clahe"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_NONE,
    STRATEGY_ROI,
    STRATEGY_ROI_BEN,
    STRATEGY_ROI_CLAHE,
    STRATEGY_ROI_BEN_CLAHE,
)

_DEFAULT_TOL: Final[int] = 7
_BEN_SIGMA_X: Final[float] = 30.0
_BEN_MASK_RADIUS_RATIO: Final[float] = 0.45
_CLAHE_CLIP_LIMIT: Final[float] = 1.0
_CLAHE_TILE_SIZE: Final[int] = 16

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


def apply_ben_graham(
    rgb: np.ndarray,
    sigma_x: float = _BEN_SIGMA_X,
    mask_radius_ratio: float = _BEN_MASK_RADIUS_RATIO,
) -> np.ndarray:
    """
    Ben Graham (Kaggle APTOS style): high-frequency emphasis.
    output = 4*img - 4*GaussianBlur(img) + 128, với mask tròn để giảm artifact viền.
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma_x)
    ben = cv2.addWeighted(rgb, 4.0, blur, -4.0, 128.0)
    h, w = ben.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(min(h, w) * mask_radius_ratio)
    cx, cy = w // 2, h // 2
    cv2.circle(mask, (cx, cy), max(r, 1), 255, -1)
    mask_3 = cv2.merge([mask, mask, mask])
    out = np.where(mask_3 > 0, ben, 128).astype(np.uint8)
    return out


def apply_clahe_lab(
    rgb: np.ndarray,
    clip_limit: float = _CLAHE_CLIP_LIMIT,
    tile_grid_size: int = _CLAHE_TILE_SIZE,
) -> np.ndarray:
    """CLAHE trên kênh L của không gian LAB (ảnh RGB uint8)."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def preprocess_roi_only(image: Image.Image) -> Image.Image:
    return extract_roi(image)


def preprocess_roi_ben(image: Image.Image) -> Image.Image:
    pil = extract_roi(image)
    arr = np.array(pil)
    out = apply_ben_graham(arr)
    return Image.fromarray(out)


def preprocess_roi_clahe(image: Image.Image) -> Image.Image:
    pil = extract_roi(image)
    arr = np.array(pil)
    out = apply_clahe_lab(arr)
    return Image.fromarray(out)


def preprocess_roi_ben_clahe(image: Image.Image) -> Image.Image:
    """Thứ tự: ROI → Ben → CLAHE (theo tài liệu quy trình)."""
    pil = extract_roi(image)
    arr = np.array(pil)
    arr = apply_ben_graham(arr)
    arr = apply_clahe_lab(arr)
    return Image.fromarray(arr)


def preprocess_none(image: Image.Image) -> Image.Image:
    return image


_PREPROCESS_FUNCS: dict[str, Callable[[Image.Image], Image.Image]] = {
    STRATEGY_NONE: preprocess_none,
    STRATEGY_ROI: preprocess_roi_only,
    STRATEGY_ROI_BEN: preprocess_roi_ben,
    STRATEGY_ROI_CLAHE: preprocess_roi_clahe,
    STRATEGY_ROI_BEN_CLAHE: preprocess_roi_ben_clahe,
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
