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

## Cấu hình Finetune

### Chiến lược học (Learning Strategy)

Toàn bộ model được fine-tune end-to-end từ pretrained ImageNet weights — không đóng băng backbone. Bài toán được mô hình hóa là **ordinal regression**: model dự đoán 1 giá trị thực trong khoảng `[0, 4]`, sau đó làm tròn và clip về nhãn nguyên.

```
Pretrained backbone (ImageNet-1k)
        ↓  fine-tune toàn bộ
Custom head: Linear(feat_dim → 1)
        ↓
output: float ∈ [0, 4]  →  round().clip(0, 4)  →  label ∈ {0,1,2,3,4}
```

### Thông số mặc định

| Thành phần       | Giá trị               | Ghi chú                                        |
|------------------|-----------------------|------------------------------------------------|
| Optimizer        | Adam                  | Không dùng weight decay                        |
| Learning Rate    | `1e-4`                | Áp dụng đồng đều toàn bộ tham số               |
| Epochs           | 15                    | Chỉnh `EPOCHS` trong Cell 2                    |
| Loss function    | `SmoothL1Loss`        | Robust với outlier hơn MSE, phù hợp ordinal    |
| Batch size       | 16 per GPU            | Effective batch = 16 × 2 GPU = 32              |
| Mixed Precision  | AMP FP16              | `torch.amp.autocast("cuda")` + `GradScaler`    |
| Parallelism      | DDP (NCCL)            | `mp.spawn` trên 2× T4, mỗi GPU 1 process       |
| Best model       | Theo **val QWK**      | Lưu tại `outputs/checkpoints/`                 |

### Thông số theo từng model

| Model            | Input size | Params  | Batch/GPU | VRAM ước tính |
|------------------|------------|---------|-----------|---------------|
| EfficientNet-B7  | 456×456    | ~66M    | 8        | ~12 GB        |
| Swin-Base        | 224×224    | ~88M    | 8        | ~10 GB        |

### Preprocessing

Không augmentation — chỉ resize và normalize bằng ImageNet statistics:

```python
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Metrics đánh giá

| Metric              | Mô tả                                              |
|---------------------|----------------------------------------------------|
| **QWK** (primary)   | Quadratic Weighted Kappa — metric chính của Kaggle |
| Accuracy            | Tỉ lệ dự đoán đúng                                 |
| Macro F1            | F1 trung bình đồng đều giữa các lớp               |
| Balanced Accuracy   | Accuracy có tính mất cân bằng lớp                 |
| Per-class Recall    | Recall riêng từng lớp 0–4                          |
| Confusion Matrix    | Ma trận nhầm lẫn (normalized theo hàng)            |
| Classification Report | Precision / Recall / F1 chi tiết mỗi lớp        |
| Train / Val Loss    | SmoothL1 loss theo từng epoch                      |

### Lưu ý khi thay đổi thông số

- **Đổi model**: thay `MODEL_TYPE` và `IMAGE_SIZE` trong Cell 2 cùng lúc
  - `"efficientnet_b7"` → `IMAGE_SIZE = 456`
  - `"swin_transformer"` → `IMAGE_SIZE = 224`
- **Tăng batch size**: kiểm tra VRAM; nếu OOM thì giảm xuống 8 hoặc dùng `gradient_accumulation`
- **Tăng epochs**: chỉnh `EPOCHS`; model sẽ tự lưu checkpoint tốt nhất theo val QWK

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
├── evaluation_metrics.txt      # Bản đầy đủ: loss, QWK, F1, recall, sklearn report
├── classification_report.txt   # Giống nội dung evaluation_metrics.txt
├── wandb_run_meta.json         # id run W&B — dùng resume sau training
└── outputs.zip                 # Tất cả được zip tự động
```

---

## Experiment Tracking (Weights & Biases)

Dùng [Weights & Biases](https://wandb.ai) cho **toàn bộ pipeline** (không dừng sau epoch cuối):

1. **Training (DDP)**: mỗi epoch — train/val loss, acc, QWK, thời gian epoch; cuối training log `pipeline/stage=training_complete`, lưu `wandb_run_meta.json`, rồi `wandb.finish()` ở worker.
2. **Notebook resume**: Cell 6–8 gọi `resume_wandb_run()` với cùng `run id` để tiếp tục log:
   - **Eval (held-out 10%)**: scalar metrics, bảng classification report (HTML), artifact `evaluation_metrics`.
   - **Submission**: phân phối nhãn trên `test.csv`, artifact `submission_csv`.
   - **Figures + outputs**: ảnh confusion matrix / curves / recall, artifact `pipeline_outputs` (cả thư mục `outputs/`), artifact `outputs_zip`; cuối cùng `wandb_finish_pipeline()` đóng run.

File `evaluation_metrics.txt` trùng format với log console (Validation Loss, QWK, Accuracy, Macro F1, Balanced Accuracy, Per-class Recall, classification report sklearn).

---

## Yêu cầu phần cứng

- 2× NVIDIA T4 GPU (Kaggle)
- CUDA 11.x hoặc mới hơn
- RAM ≥ 16GB
