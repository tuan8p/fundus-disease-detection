"""
preprocessing.py
------------------
Tiền xử lý ảnh fundus (APTOS) — deterministic, áp trước Resize + ImageNet norm trong dataset.

Thay đổi so với phiên bản cũ:
  - Bỏ ROI crop (extract_roi) vì gây inconsistency (nhiều ảnh không được crop do mask fail)
  - Thay bằng circular mask tự tính để cô lập vùng fundus, APTOS images đã căn giữa sẵn
  - Tăng sigma_x: short_side * 0.04 (min 20, max 80) thay vì 0.02 (min 10, max 60)
  - Tăng mask_radius_ratio: 0.48 thay vì 0.45 để bao phủ fundus tốt hơn, giảm halo viền
  - Tăng clip_limit CLAHE: 2.0 thay vì 1.0

Các mode:
  - none      : không tiền xử lý thêm (baseline: chỉ resize + norm trong transforms)
  - ben       : chỉ Ben Graham (circular mask)
  - clahe     : chỉ CLAHE (kênh L của LAB)
  - ben_clahe : Ben Graham → CLAHE (thứ tự theo tài liệu: Ben trước, CLAHE sau)

Không augmentation.
"""

from __future__ import annotations

from typing import Callable, Final

import cv2
import numpy as np
from PIL import Image

STRATEGY_NONE: Final[str] = "none"
STRATEGY_BEN: Final[str] = "ben"
STRATEGY_CLAHE: Final[str] = "clahe"
STRATEGY_BEN_CLAHE: Final[str] = "ben_clahe"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_NONE,
    STRATEGY_BEN,
    STRATEGY_CLAHE,
    STRATEGY_BEN_CLAHE,
)

_BEN_MASK_RADIUS_RATIO: Final[float] = 0.48   # tăng từ 0.45 → giảm halo viền
_CLAHE_CLIP_LIMIT: Final[float] = 2.0          # tăng từ 1.0 → contrast tự nhiên hơn


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def apply_ben_graham(
    rgb: np.ndarray,
    sigma_x: float | None = None,
    mask_radius_ratio: float = _BEN_MASK_RADIUS_RATIO,
) -> np.ndarray:
    """
    Ben Graham (Kaggle APTOS style): high-frequency emphasis.
    output = 4*img - 4*GaussianBlur(img) + 128, với circular mask để giảm artifact viền.

    Thay đổi:
      - sigma_x tự tính: short_side * 0.04, clamp [20, 80]  (cũ: 0.02, [10, 60])
        → blur mạnh hơn → loại noise tốt hơn trước khi subtract
      - mask_radius_ratio: 0.48  (cũ: 0.45)
        → bao phủ fundus rộng hơn, giảm viền xám lộ ra
    """
    if sigma_x is None:
        short_side = min(rgb.shape[0], rgb.shape[1])
        sigma_x = float(np.clip(short_side * 0.04, 20.0, 80.0))

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

    return np.where(mask_3 > 0, ben, 128).astype(np.uint8)


def apply_clahe_lab(
    rgb: np.ndarray,
    clip_limit: float = _CLAHE_CLIP_LIMIT,
    tile_grid_size: int | None = None,
) -> np.ndarray:
    """
    CLAHE trên kênh L của không gian LAB (ảnh RGB uint8).

    Thay đổi:
      - clip_limit mặc định: 2.0  (cũ: 1.0)
        → tăng contrast hiệu quả hơn, ít artifact hơn khi kết hợp với Ben
      - tile_grid_size tự tính: short_side // 8, clamp [8, 32]  (giữ nguyên)
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if tile_grid_size is None:
        short_side = min(rgb.shape[0], rgb.shape[1])
        tile_grid_size = int(np.clip(short_side // 8, 8, 32))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------

def preprocess_none(image: Image.Image) -> Image.Image:
    return image


def preprocess_ben(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    out = apply_ben_graham(arr)
    return Image.fromarray(out)


def preprocess_clahe(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    out = apply_clahe_lab(arr)
    return Image.fromarray(out)


def preprocess_ben_clahe(image: Image.Image) -> Image.Image:
    """Thứ tự: Ben Graham → CLAHE (theo tài liệu quy trình)."""
    arr = np.array(image.convert("RGB"))
    arr = apply_ben_graham(arr)
    arr = apply_clahe_lab(arr)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Registry & public API
# ---------------------------------------------------------------------------

_PREPROCESS_FUNCS: dict[str, Callable[[Image.Image], Image.Image]] = {
    STRATEGY_NONE: preprocess_none,
    STRATEGY_BEN: preprocess_ben,
    STRATEGY_CLAHE: preprocess_clahe,
    STRATEGY_BEN_CLAHE: preprocess_ben_clahe,
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