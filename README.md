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
    ├── models.py               # Factory tạo model (EfficientNet-B7 / Swin-Base)
    ├── train.py                # Training loop với DDP + AMP
    ├── evaluate.py             # Tính toán metrics
    ├── visualize.py            # Vẽ figures đánh giá
    └── utils.py                # Checkpoint, submission CSV, zip outputs, W&B
```

---

## Models

| Model            | Backbone                           | Input   | Pretrained  |
|------------------|------------------------------------|---------|-------------|
| EfficientNet-B7  | `tf_efficientnet_b7` (timm)        | 456×456 | ImageNet-1k |
| Swin Transformer | `swin_base_patch4_window7_224`     | 224×224 | ImageNet-1k |

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
4. Chỉnh thông số trong **Cell 2** (MODEL_TYPE, EPOCHS, BATCH_SIZE, ...)
5. Run All

---

## Thông số mặc định

| Thành phần      | Giá trị                |
|-----------------|------------------------|
| Optimizer       | Adam                   |
| LR              | 1e-4                   |
| Epochs          | 15                     |
| Loss            | SmoothL1 (regression)  |
| Batch size      | 16 per GPU             |
| Mixed Precision | AMP (FP16)             |
| GPU             | 2× T4 (DDP)            |

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
├── classification_report.txt
└── outputs.zip                 # Tất cả được zip tự động
```

---

## Experiment Tracking

Dùng [Weights & Biases](https://wandb.ai) để theo dõi:

- Train/Val loss, accuracy, QWK theo từng epoch
- Confusion matrix
- Learning rate

---

## Yêu cầu phần cứng

- 2× NVIDIA T4 GPU (Kaggle)
- CUDA 11.x hoặc mới hơn
- RAM ≥ 16GB
