
import io
import base64
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.model import build_model, predict_labels


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                      ║
# ╚══════════════════════════════════════════════════════════════╝
CHECKPOINT_EFF  = "checkpoints/best_model_efficientnet_b7.pth"
CHECKPOINT_SWIN = "checkpoints/best_model_swinv2_base_384.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Phải khớp với lúc train
IMAGE_SIZE_EFF  = 456
IMAGE_SIZE_SWIN = 384
HEAD_TYPE_EFF   = "regression"

DR_GRADE_DESCRIPTION = {
    0: "Không có dấu hiệu bệnh võng mạc tiểu đường",
    1: "Bệnh võng mạc tiểu đường nhẹ",
    2: "Bệnh võng mạc tiểu đường trung bình",
    3: "Bệnh võng mạc tiểu đường nặng",
    4: "Bệnh võng mạc tiểu đường tăng sinh (nghiêm trọng)",
}

def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Xử lý linh hoạt shape của SwinV2. 
    Tùy phiên bản timm, output có thể là 3D hoặc 4D.
    """
    if tensor.ndim == 4:
        # Nếu timm trả về [B, H, W, C] (4 giá trị)
        return tensor.permute(0, 3, 1, 2).contiguous()
    
    # Nếu timm trả về [B, seq_len, C] (3 giá trị)
    B, seq_len, C = tensor.shape
    H = W = int(seq_len ** 0.5)
    if H * W != seq_len:
        raise ValueError(
            f"seq_len={seq_len} không phải số chính phương — "
            f"kiểm tra IMAGE_SIZE_SWIN (nên là 192 hoặc 384)."
        )
    return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def _load_checkpoint(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model.to(DEVICE).eval()


def _denormalize(tensor: torch.Tensor,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)) -> np.ndarray:
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1).astype(np.float32)


def _to_base64(img_rgb_uint8: np.ndarray) -> str:
    """numpy uint8 RGB → base64 JPEG string."""
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    _, buf   = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Quản lý model sử dụng (Singleton)
class _ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def _init(self):
        if self._ready:
            return
        print(f"⏳  Loading XAI models trên {DEVICE}...")

        # ── EfficientNet-B7 ──────────────────────────────────
        self.eff_model = _load_checkpoint(
            build_model("efficientnet_b7", head_type=HEAD_TYPE_EFF, pretrained=False),
            CHECKPOINT_EFF,
        )
        self.eff_cam = GradCAM(
            model=self.eff_model,
            target_layers=[self.eff_model.backbone.features[-1]],
        )
        self.eff_tfm = transforms.Compose([
            transforms.Resize((IMAGE_SIZE_EFF, IMAGE_SIZE_EFF)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── SwinV2-Base-384 ───────────────────────────────────
        self.swin_model = _load_checkpoint(
            build_model("swinv2_base_384", pretrained=False),
            CHECKPOINT_SWIN,
        )
        self.swin_cam = EigenCAM(
            model=self.swin_model,
            target_layers=[self.swin_model.backbone.layers[-1].blocks[-1].norm1],
            reshape_transform=swin_reshape_transform,   # Đã fix lỗi 3 hoặc 4 tham số
        )
        self.swin_tfm = transforms.Compose([
            transforms.Resize((IMAGE_SIZE_SWIN, IMAGE_SIZE_SWIN)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._ready = True
        print("✅  XAI Models (Eff & Swin) sẵn sàng.")

    @classmethod
    def get(cls) -> "_ModelManager":
        obj = cls()
        obj._init()
        return obj


# ╔══════════════════════════════════════════════════════════════╗
# ║  HÀM CHÍNH                                                   ║
# ╚══════════════════════════════════════════════════════════════╝
def run_xai_for_web(image_input, target_mode: str = "all") -> dict:
    """
    Chạy XAI. target_mode có thể là 'xai-eff', 'xai-swin', hoặc 'all'
    """
    # ── Parse input ───────────────────────────────────────────
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Kiểu không hỗ trợ: {type(image_input)}")

    mgr = _ModelManager.get()
    output = {"original_b64": _pil_to_base64(pil_img)}

    # ── Chỉnh sửa danh sách configs dựa trên target_mode ──────
    configs = []
    
    if target_mode in ["xai-eff", "all"]:
        configs.append(("efficientnet", mgr.eff_model,  mgr.eff_cam,  mgr.eff_tfm,  HEAD_TYPE_EFF))
        
    if target_mode in ["xai-swin", "all"]:
        configs.append(("swinv2",       mgr.swin_model, mgr.swin_cam, mgr.swin_tfm, "regression"))

    # ── Vòng lặp chạy model ───────────────────────────────────
    for key, model, cam, tfm, head_type in configs:
        tensor = tfm(pil_img).unsqueeze(0).to(DEVICE)   # [1, 3, H, W]

        # Inference
        with torch.no_grad():
            raw_output = model(tensor)                   # [1, 1]

        grade     = int(predict_labels(raw_output, head_type=head_type).item())
        raw_score = round(float(raw_output[0].item()), 4)

        # Heatmap
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        heatmap = grayscale_cam[0]                       # [H, W]

        # Ảnh đã denormalize để overlay
        img_np = _denormalize(tensor[0])                 # [H, W, 3] float32

        # Overlay CAM lên ảnh (uint8 RGB)
        overlay_np = show_cam_on_image(img_np, heatmap, use_rgb=True)

        # Heatmap thuần — jet colormap (uint8 RGB)
        hm_norm  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_rgb   = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        output[key] = {
            "grade"      : grade,
            "raw_score"  : raw_score,
            "description": DR_GRADE_DESCRIPTION.get(grade, "Không xác định"),
            "heatmap_b64": _to_base64(hm_rgb),
            "overlay_b64": _to_base64(overlay_np),
        }

    return output

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    out  = run_xai_for_web(path, target_mode="all")
    print("Test thành công!")