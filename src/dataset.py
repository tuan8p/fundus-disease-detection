"""
dataset.py
----------
Quản lý dữ liệu APTOS 2019:
  - APTOSDataset   : Dataset class cho ảnh fundus (có/không nhãn)
  - get_transforms : Chỉ resize + normalize (ImageNet stats)
  - get_dataloaders: Stratified 3-way split (7:2:1) + DataLoader với DistributedSampler
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split

from .preprocessing import apply_preprocessing


# ── Dataset ──────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    Đọc ảnh fundus từ thư mục và trả về (image_tensor, label).
    Nếu labels=None (tập test không nhãn), trả về (image_tensor, -1).
    """

    def __init__(
        self,
        image_ids,
        image_dir,
        transform,
        labels=None,
        preprocessing_strategy: str = "none",
    ):
        """
        Args:
            image_ids (list[str]): Danh sách id_code của ảnh.
            image_dir (str)      : Đường dẫn thư mục chứa ảnh (.png).
            transform            : torchvision transforms áp dụng cho ảnh.
            labels (list[int] | None): Nhãn tương ứng; None nếu tập inference.
            preprocessing_strategy: none | roi | roi_ben | roi_imgtype | roi_ben_imgtype
        """
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels
        self.preprocessing_strategy = preprocessing_strategy

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.image_ids[idx]}.png")
        image = Image.open(img_path).convert("RGB")
        if self.preprocessing_strategy != "none":
            image = apply_preprocessing(image, self.preprocessing_strategy)
        image = self.transform(image)

        label = float(self.labels[idx]) if self.labels is not None else -1.0
        return image, torch.tensor(label, dtype=torch.float32)


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(image_size: int) -> transforms.Compose:
    """
    Trả về transform: chỉ resize về image_size×image_size rồi normalize
    bằng ImageNet mean/std. Không augmentation (baseline).

    Args:
        image_size (int): Kích thước cạnh ảnh vuông sau resize.
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


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
    preprocessing_strategy: str = "none",
):
    """
    Đọc train.csv và test.csv, tạo 4 DataLoader:
      - train_loader        : 70% của train.csv, dùng DistributedSampler khi DDP
      - val_loader          : 20% của train.csv, không shuffle
      - internal_test_loader: 10% còn lại của train.csv (held-out evaluation)
      - submit_loader       : toàn bộ test.csv (không nhãn, chỉ dùng để inference)

    Args:
        data_dir    : Thư mục gốc chứa train.csv, test.csv, train_images/, test_images/
        image_size  : Kích thước ảnh sau resize
        batch_size  : Batch size mỗi GPU
        num_workers : Số worker cho DataLoader
        train_ratio : Tỉ lệ tập train (mặc định 0.7)
        val_ratio   : Tỉ lệ tập val   (mặc định 0.2)
        seed        : Random seed để reproducibility
        rank        : Rank của tiến trình DDP (0 nếu không dùng DDP)
        world_size  : Số GPU/tiến trình DDP (1 nếu không dùng DDP)
        preprocessing_strategy: Tiền xử lý PIL trước resize (xem preprocessing.py)

    Returns:
        tuple: (train_loader, val_loader, internal_test_loader, submit_loader, train_sampler)
    """
    transform = get_transforms(image_size)

    # ── Đọc CSV ──────────────────────────────────────────────────────────────
    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_csv  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    image_ids = train_csv["id_code"].tolist()
    labels    = train_csv["diagnosis"].tolist()

    # ── Stratified 3-way split: train / val / internal_test ──────────────────
    # Bước 1: tách test (10%) ra khỏi phần còn lại
    test_size = round(1.0 - train_ratio - val_ratio, 10)  # ~0.1
    ids_trainval, ids_test, lbl_trainval, lbl_test = train_test_split(
        image_ids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Bước 2: tách val (20%) ra khỏi trainval
    # val_ratio tương đối so với tổng ban đầu → chuyển về tỉ lệ so với trainval
    relative_val = val_ratio / (train_ratio + val_ratio)
    ids_train, ids_val, lbl_train, lbl_val = train_test_split(
        ids_trainval, lbl_trainval,
        test_size=relative_val,
        stratify=lbl_trainval,
        random_state=seed,
    )

    train_dir = os.path.join(data_dir, "train_images")
    test_dir  = os.path.join(data_dir, "test_images")

    train_dataset         = APTOSDataset(
        ids_train, train_dir, transform, lbl_train, preprocessing_strategy=preprocessing_strategy
    )
    val_dataset           = APTOSDataset(
        ids_val, train_dir, transform, lbl_val, preprocessing_strategy=preprocessing_strategy
    )
    internal_test_dataset = APTOSDataset(
        ids_test, train_dir, transform, lbl_test, preprocessing_strategy=preprocessing_strategy
    )

    submit_ids    = test_csv["id_code"].tolist()
    submit_dataset = APTOSDataset(
        submit_ids, test_dir, transform, labels=None, preprocessing_strategy=preprocessing_strategy
    )

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
