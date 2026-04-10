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
STRATEGY_ROI: Final[str] = "roi"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_NONE,
    STRATEGY_BEN,
    STRATEGY_CLAHE,
    STRATEGY_ROI,
)

_BEN_MASK_RADIUS_RATIO: Final[float] = 0.48   # tăng từ 0.45 → giảm halo viền
_CLAHE_CLIP_LIMIT: Final[float] = 2.0          # tăng từ 1.0 → contrast tự nhiên hơn


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def apply_ben_graham(rgb: np.ndarray, sigma_x: float | None = None) -> np.ndarray:
    """
    Ben Graham chuẩn y khoa: Không dùng vòng tròn cứng.
    Dùng mặt nạ động (Dynamic Mask) để giữ nguyên vẹn hình dáng võng mạc thật.
    """
    if sigma_x is None:
        short_side = min(rgb.shape[0], rgb.shape[1])
        sigma_x = float(np.clip(short_side * 0.04, 20.0, 80.0))

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # 1. Tạo mặt nạ động dựa trên điểm ảnh thực (loại bỏ padding đen)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 7  # Ngưỡng giống với extract_roi
    mask_3 = np.stack([mask, mask, mask], axis=-1)

    # 2. Pre-fill nền thành 128 TRƯỚC khi Blur để tránh halo artifact viền
    #    Vấn đề: GaussianBlur pha trộn pixel viền sáng (200) với nền đen (0)
    #    → blur ≈ 100 → ben = 4×200 - 4×100 + 128 = 528 → viền trắng chói
    #    Fix: nếu nền = 128 → blur ≈ (200+128)/2 = 164 → ben ≈ 128 → trung tính
    rgb_filled = np.where(mask_3, rgb, 128).astype(np.uint8)

    # 3. Áp dụng công thức Ben Graham trên ảnh đã fill nền
    blur = cv2.GaussianBlur(rgb_filled, (0, 0), sigmaX=sigma_x)
    ben = cv2.addWeighted(rgb_filled, 4.0, blur, -4.0, 128.0)

    # 4. Phủ lại nền xám (128) cho phần ngoài võng mạc (đảm bảo sạch hoàn toàn)
    return np.where(mask_3, ben, 128).astype(np.uint8)

def apply_clahe_lab(
    rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int | None = None,
) -> np.ndarray:
    """
    CLAHE trên kênh L. Bắt buộc phải xóa sạch artifact ở viền padding đen sau khi chạy.
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # 1. Tạo mặt nạ động để ghi nhớ vị trí viền đen
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    mask_3 = np.stack([mask, mask, mask], axis=-1)

    if tile_grid_size is None:
        short_side = min(rgb.shape[0], rgb.shape[1])
        tile_grid_size = int(np.clip(short_side // 8, 8, 32))

    # 2. Áp dụng CLAHE
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    clahe_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # 3. Ép vùng background về màu đen tuyệt đối (0) để xóa artifact
    return np.where(mask_3, clahe_rgb, 0).astype(np.uint8)

def extract_roi(image: Image.Image, tol: int = 7, pad_ratio: float = 0.05) -> Image.Image:
    """
    Cắt vùng fundus (chỉ lấy khối sáng lớn nhất), bỏ viền đen, 
    pad nhẹ rồi căn vuông (letterbox đen).
    """
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # 1. Tạo mask cơ bản
    _, mask = cv2.threshold(gray, tol, 255, cv2.THRESH_BINARY)
    
    # 2. Tìm các contour (đường viền) trong mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu ảnh đen thui hoặc lỗi, trả về ảnh gốc
    if not contours:
        return image
        
    # 3. LỌC NHIỄU CHÍ MẠNG: Chỉ lấy Contour có diện tích lớn nhất (nhãn cầu)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 4. Thêm padding để không cắt phạm rìa võng mạc (logic của bạn, rất tốt)
    pad_y = max(1, int(h * pad_ratio))
    pad_x = max(1, int(w * pad_ratio))
    
    y0 = max(0, y - pad_y)
    x0 = max(0, x - pad_x)
    y1 = min(arr.shape[0], y + h + pad_y)
    x1 = min(arr.shape[1], x + w + pad_x)
    
    cropped = arr[y0:y1, x0:x1].copy()
    
    # 5. Ép về hình vuông 1:1 (Pad-to-square)
    ch, cw = cropped.shape[0], cropped.shape[1]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off : y_off + ch, x_off : x_off + cw] = cropped
    
    return Image.fromarray(square)
# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------

def preprocess_none(image: Image.Image) -> Image.Image:
    return image


def preprocess_ben(image: Image.Image) -> Image.Image:
    # Cắt gọt vuông vức trước -> Rồi mới đưa vào Ben Graham
    cropped_img = extract_roi(image)
    arr = np.array(cropped_img.convert("RGB"))
    out = apply_ben_graham(arr)
    return Image.fromarray(out)

def preprocess_clahe(image: Image.Image) -> Image.Image:
    # Cắt gọt vuông vức trước -> Rồi mới đưa vào CLAHE
    cropped_img = extract_roi(image)
    arr = np.array(cropped_img.convert("RGB"))
    out = apply_clahe_lab(arr)
    return Image.fromarray(out)

def preprocess_roi(image: Image.Image) -> Image.Image:
    return extract_roi(image)


# ---------------------------------------------------------------------------
# Registry & public API
# ---------------------------------------------------------------------------

_PREPROCESS_FUNCS: dict[str, Callable[[Image.Image], Image.Image]] = {
    STRATEGY_NONE: preprocess_none,
    STRATEGY_BEN: preprocess_ben,
    STRATEGY_CLAHE: preprocess_clahe,
    STRATEGY_ROI: preprocess_roi,
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