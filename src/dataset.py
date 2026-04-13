import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

from .augmentation import (
    get_train_transforms_v1_basic,
    get_train_transforms_v2_advanced,
    get_train_transforms_v3_extreme,
    get_valid_transforms
)

# ── Dataset ──────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    def __init__(
        self,
        image_ids,
        image_dir,
        transform,
        labels=None,
        preprocessing_strategy: str = "none",
    ):
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
        
      
        image_np = np.array(image)
        if self.transform is not None:
            
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            
            from torchvision.transforms import functional as TF
            image_tensor = TF.to_tensor(image_np)

        label = float(self.labels[idx]) if self.labels is not None else -1.0
        return image_tensor, torch.tensor(label, dtype=torch.float32)


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    aug_version: str = "v1",
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    preprocessing_strategy: str = "none",
):
    if aug_version == "v1":
        train_transform = get_train_transforms_v1_basic(image_size)
    elif aug_version == "v2":
        train_transform = get_train_transforms_v2_advanced(image_size)
    elif aug_version == "v3":
        train_transform = get_train_transforms_v3_extreme(image_size)
    else:
        train_transform = get_train_transforms_v1_basic(image_size)
        
    valid_transform = get_valid_transforms(image_size)

    # ── Đọc CSV ──────────────────────────────────────────────────────────────
    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_csv  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    image_ids = train_csv["id_code"].tolist()
    labels    = train_csv["diagnosis"].tolist()

    # ── Stratified 3-way split ───────────────────────────────────────────────
    test_size = round(1.0 - train_ratio - val_ratio, 10)
    ids_trainval, ids_test, lbl_trainval, lbl_test = train_test_split(
        image_ids, labels, test_size=test_size, stratify=labels, random_state=seed,
    )

    relative_val = val_ratio / (train_ratio + val_ratio)
    ids_train, ids_val, lbl_train, lbl_val = train_test_split(
        ids_trainval, lbl_trainval, test_size=relative_val, stratify=lbl_trainval, random_state=seed,
    )

    train_dir = os.path.join(data_dir, "train_images")
    test_dir  = os.path.join(data_dir, "test_images")

    train_dataset = APTOSDataset(ids_train, train_dir, train_transform, lbl_train, preprocessing_strategy)
    val_dataset   = APTOSDataset(ids_val, train_dir, valid_transform, lbl_val, preprocessing_strategy)
    internal_test_dataset = APTOSDataset(ids_test, train_dir, valid_transform, lbl_test, preprocessing_strategy)

    submit_ids = test_csv["id_code"].tolist()
    submit_dataset = APTOSDataset(submit_ids, test_dir, valid_transform, labels=None, preprocessing_strategy=preprocessing_strategy)

    # ── QUẢN LÝ LUỒNG SAMPLER  ───────────────────────────────────────────────
    use_ddp = world_size > 1
    
    # Tính toán weights chung cho training 
    class_counts = np.bincount(lbl_train)
    class_weights = 1.0 / (class_counts + 1e-5) # Nghịch đảo tần suất
    sample_weights_np = np.array([class_weights[int(l)] for l in lbl_train])
    sample_weights = torch.from_numpy(sample_weights_np).double()

    if use_ddp:
        # DistributedWeightedSampler
        import math
        class DistributedWeightedSampler(DistributedSampler):
            def __init__(self, dataset, weights, num_replicas=None, rank=None,
                         replacement=True, seed=0):
                super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                                 shuffle=False, seed=seed)
                self.weights = weights
                self.replacement = replacement

            def __iter__(self):
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)

                # 1. Sample đúng len(weights) indices có trọng số
                indices = torch.multinomial(
                    self.weights,
                    len(self.weights),
                    self.replacement,
                    generator=g,
                ).tolist()

                # 2. Pad để chia hết cho num_replicas
                if len(indices) < self.total_size:
                    repeats = math.ceil(self.total_size / len(indices))
                    indices = (indices * repeats)[:self.total_size]

                # 3. Mỗi GPU chỉ lấy phần của mình
                indices = indices[self.rank : self.total_size : self.num_replicas]
                assert len(indices) == self.num_samples
                return iter(indices)

        train_sampler = DistributedWeightedSampler(
            train_dataset, 
            weights=sample_weights,
            num_replicas=world_size, 
            rank=rank
        )
    else:
        # Chạy Local -> KÍCH HOẠT WEIGHTED RANDOM SAMPLER
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    # ── Khởi tạo DataLoader ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    internal_test_loader = DataLoader(internal_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    submit_loader = DataLoader(submit_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, internal_test_loader, submit_loader, train_sampler