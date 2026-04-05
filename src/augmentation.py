"""
augmentation.py
---------------
Tái hiện bộ Data Augmentation của Top 2 Kaggle APTOS 2019.
Phục vụ Ablation Study: V1 (Cơ bản) -> V2 (Nâng cao) -> V3 (Cực đoan).
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================================================================
# V1: HÌNH HỌC CƠ BẢN (BASIC SPATIAL)
# Chỉ thay đổi góc nhìn, an toàn tuyệt đối 100%.
# Gồm: Flip (Lật), Transpose (Chuyển vị), ShiftScaleRotate (Dịch-Phóng-Xoay).
# =====================================================================
def get_train_transforms_v1_basic(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=90, p=0.7),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# =====================================================================
# V2: NHIỄU & MÀU SẮC (ADVANCED NOISE & COLOR)
# V1 + Thay đổi môi trường chụp + Che khuất thông minh.
# Thêm: BrightnessContrast, HueSaturation, Blur, CoarseDropout.
# =====================================================================
def get_train_transforms_v2_advanced(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # 1. Hình học (Kế thừa V1)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=90, p=0.7),
        
        # 2. Màu sắc & Ánh sáng
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=0.3),
        
        # 3. Nhiễu & Che khuất
        A.Blur(blur_limit=3, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# =====================================================================
# V3: BIẾN DẠNG CỰC ĐOAN (EXTREME DISTORTION - THE KAGGLE WAY)
# V2 + Bóp méo cao su + Cân bằng sáng cục bộ.
# Thêm: ElasticTransform, GridDistortion, CLAHE.
# =====================================================================
def get_train_transforms_v3_extreme(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # 1. Hình học
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=90, p=0.7),
        
        # 2. Bóp méo hình học (Vũ khí bí mật nhưng nguy hiểm)
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3), # Chỉ cho 30% ảnh bị bóp méo để tránh hỏng dữ liệu
        
        # 3. Màu sắc & Ánh sáng
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3), # CLAHE chạy on-the-fly
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=0.3),
        
        # 4. Nhiễu & Che khuất
        A.Blur(blur_limit=3, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# =====================================================================
# VALID/TEST: KHÔNG BAO GIỜ THAY ĐỔI
# =====================================================================
def get_valid_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])