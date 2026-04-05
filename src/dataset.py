"""
dataset.py
----------
Quản lý dữ liệu APTOS 2019:
  - Đã tích hợp Albumentations (Dùng OpenCV thay vì PIL)
  - Phân tách rõ ràng luồng Transform cho Train và Valid
  - get_dataloaders: Stratified 3-way split (7:2:1) + DataLoader
"""

import os
import cv2 # thay thế PIL bằng cv2
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

# ── IMPORT BỘ AUGMENTATION TỪ FILE CỦA BẢO ──────────────────────────────────
from src.augmentation import (
    get_train_transforms_v1_basic,
    get_train_transforms_v2_advanced,
    get_train_transforms_v3_extreme,
    get_valid_transforms
)

# ── Dataset ──────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    Đọc ảnh fundus từ thư mục và trả về (image_tensor, label).
    Nếu labels=None (tập test không nhãn), trả về (image_tensor, -1).
    """

    def __init__(self, image_ids, image_dir, transform, labels=None):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.image_ids[idx]}.png")
        
        # 1. Dùng OpenCV đọc ảnh (Yêu cầu bắt buộc của Albumentations)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")
            
        # 2. Chuyển hệ màu từ BGR (mặc định của cv2) sang RGB chuẩn
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Áp dụng Augmentation (Cú pháp chuẩn của Albumentations)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = float(self.labels[idx]) if self.labels is not None else -1.0
        return image, torch.tensor(label, dtype=torch.float32)

# (ĐÃ XÓA hàm get_transforms của torchvision vì không còn dùng nữa)

# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
):
    
    # =====================================================================
    # CHỌN PHIÊN BẢN AUGMENTATION ĐỂ CHẠY (ABLATION STUDY)
    # Bạn chỉ cần mở comment (uncomment) bản muốn test, đóng các bản còn lại.
    # =====================================================================
    # train_transform = get_train_transforms_v1_basic(image_size)
    train_transform = get_train_transforms_v2_advanced(image_size) # Đang bật V2
    # train_transform = get_train_transforms_v3_extreme(image_size)
    
    # Băng chuyền cố định không ngẫu nhiên cho Validation / Test
    valid_transform = get_valid_transforms(image_size)
    # =====================================================================

    # ── Đọc CSV ──────────────────────────────────────────────────────────────
    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_csv  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    image_ids = train_csv["id_code"].tolist()
    labels    = train_csv["diagnosis"].tolist()

    # ── Stratified 3-way split ───────────────────────────────────────────────
    test_size = round(1.0 - train_ratio - val_ratio, 10)  # ~0.1
    ids_trainval, ids_test, lbl_trainval, lbl_test = train_test_split(
        image_ids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    relative_val = val_ratio / (train_ratio + val_ratio)
    ids_train, ids_val, lbl_train, lbl_val = train_test_split(
        ids_trainval, lbl_trainval,
        test_size=relative_val,
        stratify=lbl_trainval,
        random_state=seed,
    )

    train_dir = os.path.join(data_dir, "train_images")
    test_dir  = os.path.join(data_dir, "test_images")

    # ── GÁN TRANSFORM TƯƠNG ỨNG CHO TỪNG TẬP DỮ LIỆU ─────────────────────────
    # Chỉ định tập Train xài bộ Augmentation biến hóa
    train_dataset = APTOSDataset(ids_train, train_dir, train_transform, lbl_train)
    
    # Valid, Internal Test, Submit bắt buộc xài bộ chuẩn hóa (Valid)
    val_dataset           = APTOSDataset(ids_val,   train_dir, valid_transform, lbl_val)
    internal_test_dataset = APTOSDataset(ids_test,  train_dir, valid_transform, lbl_test)
    submit_ids    = test_csv["id_code"].tolist()
    submit_dataset = APTOSDataset(submit_ids, test_dir, valid_transform, labels=None)

    # ── DistributedSampler cho DDP (chỉ train) ───────────────────────────────
    use_ddp = world_size > 1
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if use_ddp else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    internal_test_loader = DataLoader(
        internal_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    submit_loader = DataLoader(
        submit_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, internal_test_loader, submit_loader, train_sampler