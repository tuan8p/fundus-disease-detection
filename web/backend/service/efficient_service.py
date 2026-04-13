import io
import torch
from PIL import Image
from torchvision import transforms


from model.model import build_model, predict_labels

# ── Config ──────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/best_model_efficientnet_b7.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 456

HEAD_TYPE = "regression" 

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Load model 1 lần duy nhất ───────────────────────────────
def load_effb7_model():
    model = build_model(
        model_type="efficientnet_b7",
        head_type=HEAD_TYPE,
        pretrained=False,
        freeze_strategy="none",
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Lấy state_dict, hỗ trợ cả trường hợp lưu nguyên dict hoặc lưu dạng dict{"model_state": ...}
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

    # Xử lý prefix nếu lúc train dùng DataParallel (module.)
    first_key = list(state_dict.keys())[0]
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # KHÁC VỚI SWIN: Không tự động chèn "backbone." vào nữa.
    # Vì FundusRegressor của bạn có cả "self.backbone" và "self.head", 
    # checkpoint chuẩn khi lưu đã có sẵn các tiền tố "backbone." và "head." rồi.

    # Nạp weights vào model
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ EfficientNet-B7 loaded on {DEVICE}")
    return model

# Biến global giữ model
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_effb7_model()
    return _model


# ── Hàm predict chính ───────────────────────────────────────
DR_GRADE_DESCRIPTION = {
    0: "Không có dấu hiệu bệnh võng mạc tiểu đường",
    1: "Bệnh võng mạc tiểu đường nhẹ",
    2: "Bệnh võng mạc tiểu đường trung bình",
    3: "Bệnh võng mạc tiểu đường nặng",
    4: "Bệnh võng mạc tiểu đường tăng sinh (nghiêm trọng)",
}

def predict_fundus_effb7(image_bytes: bytes) -> dict:
    model = get_model()

    # 1. Preprocess
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, SIZE, SIZE]

    # 2. Inference
    with torch.no_grad():
        raw_output = model(tensor)  # [1, 1] hoặc [1, 4] tùy thuộc regression/ordinal

    # 3. Convert sang label
    # Cần truyền đúng head_type để hàm dự đoán xử lý chuẩn xác
    label = predict_labels(raw_output, head_type=HEAD_TYPE)
    grade = int(label.item())
    
    # Tính raw_score (nếu là regression thì lấy trực tiếp, ordinal thì có thể bỏ qua hoặc xử lý khác)
    score = round(float(raw_output[0].item()), 4) if HEAD_TYPE == "regression" else None

    return {
        "grade": grade,
        "raw_score": score,
        "description": DR_GRADE_DESCRIPTION.get(grade, "Không xác định"),
    }