# Fundus Disease Detection — APTOS 2019 Baseline

Baseline phát hiện bệnh võng mạc tiểu đường (Diabetic Retinopathy) cho cuộc thi [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) trên Kaggle.

## Bài toán

Phân loại mức độ bệnh võng mạc tiểu đường từ ảnh fundus, 5 lớp (0–4):

| Label | Mô tả                        |
|-------|------------------------------|
| 0     | No DR (Bình thường)          |
| 1     | Mild DR                      |
| 2     | Moderate DR                  |
| 3     | Severe DR                    |
| 4     | Proliferative DR             |

**Metric chính**: Quadratic Weighted Kappa (QWK)

---

## Cấu trúc Repo

```
fundus-disease-detection/
├── README.md
├── requirements.txt
├── notebooks/
│   └── baseline.ipynb          # Notebook chính — chạy toàn bộ pipeline
└── src/
    ├── __init__.py
    ├── dataset.py              # Dataset, transforms, dataloaders
    ├── models.py               # Factory tạo model + freeze strategy
    ├── train.py                # Training loop với DDP + AMP + scheduler
    ├── evaluate.py             # Tính toán metrics
    ├── visualize.py            # Vẽ figures đánh giá
    ├── xai.py                  # Explainability (GradCAM, v.v.)
    └── utils.py                # Checkpoint, submission CSV, zip outputs, W&B
```

---

## Models

| Model            | Backbone                                          | Input   | Pretrained   |
|------------------|---------------------------------------------------|---------|--------------|
| EfficientNet-B7  | `tf_efficientnet_b7` (timm)                       | 456×456 | ImageNet-1k  |
| SwinV2-Base      | `swinv2_base_window12to24_192to384_22kft1k` (timm)| 384×384 | ImageNet-22k |

Pretrained weights tải tự động từ HuggingFace Hub qua `timm`.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy (trên Kaggle)

1. Upload toàn bộ repo lên Kaggle (dùng Kaggle Dataset hoặc GitHub integration)
2. Thêm `WANDB_API_KEY` vào **Kaggle Secrets** (Add-ons → Secrets)
3. Mở `notebooks/baseline.ipynb`
4. Chỉnh thông số trong **Cell 2** (xem bảng bên dưới)
5. Run All

---

## Cấu hình Finetune

### Chiến lược học (Learning Strategy)

Bài toán được mô hình hóa là **Ordinal Regression**: model dự đoán 1 giá trị thực trong khoảng `[0, 4]`, sau đó làm tròn và clip về nhãn nguyên.

```
Pretrained backbone (ImageNet)
        ↓  fine-tune với Differential LR
Custom head: Linear(feat_dim → 1)
        ↓
output: float ∈ [0, 4]  →  round().clip(0, 4)  →  label ∈ {0,1,2,3,4}
```

### Thông số mặc định (Cell 2 — notebook)

| Thành phần        | EfficientNet-B7         | SwinV2-Base                         | Ghi chú |
|-------------------|-------------------------|-------------------------------------|---------|
| `MODEL_TYPE`      | `"efficientnet_b7"`     | `"swinv2_base_384"`                 | Chỉnh trong Cell 2 |
| `IMAGE_SIZE`      | `456`                   | `384`                               | Phải khớp model |
| `LR`              | `1e-4`                  | `2e-5` → `5e-5`                     | Head LR; backbone = LR × 0.1 |
| `EPOCHS`          | `20`                    | `20`                                | |
| `BATCH_SIZE`      | `8` / GPU               | `8` / GPU                           | Effective = 8 × số GPU |
| `FREEZE_STRATEGY` | `"none"`                | `"none"` / `"partial"`              | Xem mục bên dưới |
| `CLASS_WEIGHTS`   | auto-computed           | auto-computed                       | Tính từ phân phối APTOS |
| `LR_FACTOR`       | `0.5`                   | —                                   | ReduceLROnPlateau |
| `LR_PATIENCE`     | `3`                     | —                                   | ReduceLROnPlateau |
| `WARMUP_EPOCHS`   | —                       | `2`                                 | Linear warmup cho SwinV2 |
| `T_0`             | —                       | `None` (= EPOCHS − WARMUP_EPOCHS)   | Chu kỳ Cosine |
| `USE_AMP`         | `True`                  | `True`                              | Mixed Precision FP16 |

### Optimizer — AdamW + Differential Learning Rate

Học riêng biệt cho backbone và head để bảo toàn pretrained weights:

```
backbone params  →  lr = LR × 0.1   (nhỏ hơn 10×)
head params      →  lr = LR
weight_decay     →  1e-5 (EfficientNet)  |  0.05 (SwinV2, bắt buộc để tránh overfit)
```

### Loss Function — WeightedSmoothL1Loss

Custom loss xử lý mất cân bằng nhãn trong APTOS 2019:

```python
# Phân phối APTOS 2019: Class 0 chiếm ~49%, Class 3 chỉ ~5%
_class_counts = [1805, 370, 999, 193, 295]
CLASS_WEIGHTS = [n_samples / (n_classes × count) for count in _class_counts]
# ≈ [0.40, 1.97, 0.73, 3.80, 2.48]
```

- Loss = `smooth_l1(pred, target, reduction='none') × weight[target_class]`
- Nhãn hiếm (class 3, 4) có weight cao → model bị phạt nặng hơn → kéo per-class recall và QWK lên
- Nếu không truyền `CLASS_WEIGHTS` vào CFG → fallback về `SmoothL1Loss` thuần tuý

### Scheduler theo kiến trúc

| Kiến trúc        | Scheduler                                             | Lý do |
|------------------|-------------------------------------------------------|-------|
| EfficientNet (CNN)| `ReduceLROnPlateau(mode="max", patience=3)`           | Theo dõi `val_qwk` trực tiếp, an toàn, không cần warmup |
| SwinV2 (Transformer)| `LinearLR (warmup) → CosineAnnealingWarmRestarts`  | Transformer cần warmup từ LR×0.01→LR để ổn định attention |

### Freeze Strategy (`FREEZE_STRATEGY`)

| Giá trị     | Hành vi                                                  | Dùng khi                    |
|-------------|----------------------------------------------------------|-----------------------------|
| `"none"`    | Train toàn bộ network (end-to-end)                       | Mặc định, có đủ dữ liệu     |
| `"head_only"` | Đóng băng backbone, chỉ train classifier head          | Linear Probing, dữ liệu ít  |
| `"partial"` | Unfreeze block/stage cuối + head; đóng băng phần còn lại | Fine-tune tiết kiệm VRAM    |

### Preprocessing

Không augmentation — chỉ resize và normalize bằng ImageNet statistics:

```python
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Metrics đánh giá

| Metric                | Mô tả                                              |
|-----------------------|----------------------------------------------------|
| **QWK** (primary)     | Quadratic Weighted Kappa — metric chính của Kaggle |
| Accuracy              | Tỉ lệ dự đoán đúng                                 |
| Macro F1              | F1 trung bình đồng đều giữa các lớp               |
| Balanced Accuracy     | Accuracy có tính mất cân bằng lớp                 |
| Per-class Recall      | Recall riêng từng lớp 0–4                          |
| Confusion Matrix      | Ma trận nhầm lẫn (normalized theo hàng)            |
| Classification Report | Precision / Recall / F1 chi tiết mỗi lớp          |
| Train / Val Loss      | WeightedSmoothL1 loss theo từng epoch              |
| `lr/head` & `lr/backbone` | Learning rate thực tế mỗi epoch (W&B)        |

### Lưu ý khi thay đổi thông số

- **Đổi model**: thay `MODEL_TYPE` và `IMAGE_SIZE` đồng thời
  - `"efficientnet_b7"` → `IMAGE_SIZE = 456`, `LR = 1e-4`
  - `"swinv2_base_384"` → `IMAGE_SIZE = 384`, `LR = 2e-5`
- **CLASS_WEIGHTS**: tính tự động từ `_class_counts` trong Cell 2; cập nhật nếu dùng dataset khác
- **Tăng batch size**: kiểm tra VRAM; nếu OOM thì giảm xuống 4
- **Tăng epochs**: chỉnh `EPOCHS`; model tự lưu checkpoint tốt nhất theo val QWK

---

## Data Split

```
train.csv (3,662 ảnh có nhãn) — stratified split
    ├── 70% Train  (~2,563 ảnh)
    ├── 20% Val    (~732 ảnh)   ← lưu best model (val QWK)
    └── 10% Test   (~366 ảnh)  ← held-out evaluation

test.csv (1,929 ảnh không nhãn) ← inference → submission.csv
```

---

## Output

```
outputs/
├── checkpoints/best_model_{model_type}.pth
├── figures/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── per_class_recall.png
├── submission.csv              # Submit lên Kaggle
├── evaluation_metrics.txt      # loss, QWK, F1, recall, sklearn report
├── classification_report.txt
├── wandb_run_meta.json         # id run W&B — dùng resume sau training
├── pipeline_full_console.log   # Toàn bộ stdout/stderr notebook + log DDP rank 0
└── outputs.zip                 # Tất cả được zip tự động
```

---

## Experiment Tracking (Weights & Biases)

Dùng [Weights & Biases](https://wandb.ai) cho **toàn bộ pipeline**:

1. **Training (DDP)**: mỗi epoch log `train/loss`, `val/loss`, `train/qwk`, `val/qwk`, **`lr/head`**, **`lr/backbone`**, `epoch_time_s`; cuối training log `pipeline/stage=training_complete`, lưu `wandb_run_meta.json`.
2. **Notebook resume** (Cell 6–8): tiếp tục cùng run id để log:
   - **Eval (held-out 10%)**: scalar metrics, bảng classification report (HTML), artifact `evaluation_metrics`.
   - **Submission**: phân phối nhãn trên `test.csv`, artifact `submission_csv`.
   - **Figures + outputs**: confusion matrix / curves / recall, artifact `pipeline_outputs`, artifact `outputs_zip`.

### Log console đầy đủ (`pipeline_full_console.log`)

- **Cell 2** gọi `start_pipeline_console_capture(OUTPUT_DIR)` — mọi `print` / stderr của notebook được **tee** vào file.
- **DDP**: rank 0 trong `train.py` **append** từng dòng epoch, best model, kết thúc training vào cùng file.
- **Cell 8**: `stop_pipeline_console_capture()` đóng file, W&B upload artifact `pipeline_console_log`.

---

## Yêu cầu phần cứng

| Thành phần | Yêu cầu                    |
|------------|----------------------------|
| GPU        | 2× NVIDIA T4 (Kaggle)      |
| VRAM       | ~12 GB / GPU (B7), ~14 GB / GPU (SwinV2-384) |
| CUDA       | 11.x hoặc mới hơn          |
| RAM        | ≥ 16 GB                    |
