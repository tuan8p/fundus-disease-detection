import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms_v1(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Lật ảnh
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Gom chung Scale (0.2), Shift (0.2), Rotate (180), Shear (0.2) vào 1 hàm Affine
        A.Affine(
            scale=(0.8, 1.2),           # Phóng to/thu nhỏ +- 20%
            translate_percent=0.2,      # Dịch chuyển +- 20%
            rotate=(-180, 180),         # Xoay từ -180 đến 180 độ
            shear=(-20, 20),            # Kéo lệch +- 20 độ
            p=0.8                       # Xác suất áp dụng 80%
        ),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_train_transforms_v2(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # --- 1. SPATIAL ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=0.2,
            rotate=(-180, 180),
            shear=(-20, 20),
            p=0.8
        ),
        
        # --- 2. COLOR & LIGHTING  ---
        # contrast_range=0.2, brightness_range=20
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        
        # hue_range=10., saturation_range=20.
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=20, 
            val_shift_limit=0, # Giữ nguyên Value vì đã dùng Brightness ở trên
            p=0.5
        ),
        
        # blur_and_sharpen=True
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),                  # Làm mờ nhẹ
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0) # Làm sắc nét
        ], p=0.3), # Có 30% xác suất sẽ bị mờ hoặc sắc nét
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_valid_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])