import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.model import build_model, predict_labels


# ── Config ──────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/best_model_swinv2_base_384.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Load model 1 lần duy nhất ───────────────────────────────
def load_swin_model():
    model = build_model(
        model_type="swinv2_base_384",
        pretrained=False,
        freeze_strategy="none",
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state"]

    # Debug: xem key đầu tiên
    first_key = list(state_dict.keys())[0]
    print(f"🔍 Key đầu tiên trong checkpoint: {first_key}")

    # Xử lý prefix
    if first_key.startswith("module."):
        # DataParallel → bỏ "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        first_key = list(state_dict.keys())[0]

    if not first_key.startswith("backbone."):
        # Checkpoint lưu không có wrapper → thêm "backbone."
        state_dict = {"backbone." + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"✅ SwinV2 loaded on {DEVICE}")
    return model

# Biến global giữ model trong suốt vòng đời server
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_swin_model()
    return _model


# ── Hàm predict chính ───────────────────────────────────────
DR_GRADE_DESCRIPTION = {
    0: "Không có dấu hiệu bệnh võng mạc tiểu đường",
    1: "Bệnh võng mạc tiểu đường nhẹ",
    2: "Bệnh võng mạc tiểu đường trung bình",
    3: "Bệnh võng mạc tiểu đường nặng",
    4: "Bệnh võng mạc tiểu đường tăng sinh (nghiêm trọng)",
}

def predict_fundus(image_bytes: bytes) -> dict:
    model = get_model()

    # 1. Preprocess
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 384, 384]

    # 2. Inference
    with torch.no_grad():
        raw_output = model(tensor)  # [1, 1]

    # 3. Convert sang label
    label = predict_labels(raw_output, head_type="regression")
    grade = int(label.item())
    score = round(float(raw_output.item()), 4)

    return {
        "grade": grade,
        "raw_score": score,
        "description": DR_GRADE_DESCRIPTION[grade],
    }