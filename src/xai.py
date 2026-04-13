
import io
import base64
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models import build_model, predict_labels


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                      ║
# ╚══════════════════════════════════════════════════════════════╝
CHECKPOINT_EFF  = "checkpoints/best_model_efficientnet_b7.pth"
CHECKPOINT_SWIN = "checkpoints/best_model_swinv2_base_384.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()

    # Phổ biến hơn: [B, seq_len, C]
    B, seq_len, C = tensor.shape
    H = W = int(seq_len ** 0.5)
    if H * W != seq_len:
        raise ValueError(
            f"seq_len={seq_len} không phải số chính phương. "
            f"Kiểm tra IMAGE_SIZE_SWIN (nên là 192 hoặc 384)."
        )
    return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def _load_checkpoint(model, ckpt_path: str):
    ckpt       = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model.to(DEVICE).eval()


def _denormalize(
    tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """Tensor ImageNet-normalized → numpy float32 RGB [0, 1]."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * np.array(std) + np.array(mean), 0, 1).astype(np.float32)


def _to_base64(img_rgb_uint8: np.ndarray) -> str:
    """numpy uint8 RGB → base64 JPEG."""
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    _, buf   = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_image_input(image_input) -> Image.Image:
    """Chấp nhận bytes | str path | PIL.Image → PIL.Image RGB."""
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    raise TypeError(f"Kiểu không hỗ trợ: {type(image_input)}")


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
        self.swin_model = _load_checkpoint(
            build_model("swinv2_base_384", pretrained=False),
            CHECKPOINT_SWIN,
        )
        self.swin_cam = EigenCAM(
            model=self.swin_model,
            target_layers=[self.swin_model.backbone.layers[-1].blocks[-1].norm1],
            reshape_transform=swin_reshape_transform,
        )
        self.swin_tfm = transforms.Compose([
            transforms.Resize((IMAGE_SIZE_SWIN, IMAGE_SIZE_SWIN)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._ready = True
        print("✅  XAI Models sẵn sàng.")

    @classmethod
    def get(cls) -> "_ModelManager":
        obj = cls()
        obj._init()
        return obj

def run_single_model(
    pil_img: Image.Image,
    model,
    cam,
    tfm,
    head_type: str,
) -> dict:
    """
    Trả về:
      grade, raw_score, description,
      heatmap_b64 (jet colormap thuần),
      overlay_b64 (heatmap chồng lên ảnh)
    """
    tensor = tfm(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_output = model(tensor)

    grade     = int(predict_labels(raw_output, head_type=head_type).item())
    raw_score = round(float(raw_output[0].item()), 4)

    heatmap = cam(input_tensor=tensor, targets=None)[0]

    #Overlay
    img_np     = _denormalize(tensor[0])
    overlay_np = show_cam_on_image(img_np, heatmap, use_rgb=True)

    #Heatmap
    hm_norm  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_rgb   = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    return {
        "grade"      : grade,
        "raw_score"  : raw_score,
        "description": DR_GRADE_DESCRIPTION.get(grade, "Không xác định"),
        "heatmap_b64": _to_base64(hm_rgb),
        "overlay_b64": _to_base64(overlay_np),
    }