"""
models.py
---------
Factory tạo model cho bài toán ordinal regression (DR grading).

Hai model được hỗ trợ:
  - "efficientnet_b7"  : tf_efficientnet_b7, input 456×456
  - "swin_transformer" : swin_base_patch4_window7_224, input 224×224

Cả hai đều:
  - Load pretrained ImageNet-1k weights từ HuggingFace Hub (qua timm)
  - Thay classifier head thành 1 output neuron (ordinal regression)
  - Dễ mở rộng: thêm model mới chỉ cần thêm vào REGISTRY
"""

import torch
import torch.nn as nn
import timm


# ── Registry: tên model → timm backbone name ─────────────────────────────────

REGISTRY = {
    "efficientnet_b7": "tf_efficientnet_b7",
    "swin_transformer": "swin_base_patch4_window7_224",
}


# ── Model wrapper ─────────────────────────────────────────────────────────────

class FundusRegressor(nn.Module):
    """
    Wrapper bọc backbone timm, thay head thành 1 neuron cho ordinal regression.

    Output: tensor shape [B, 1] — giá trị thực trong khoảng ~[0, 4].
    Inference: round(output).clip(0, 4) → nhãn nguyên 0–4.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone đã có head 1 neuron, trả về [B, 1]
        return self.backbone(x)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(model_type: str, pretrained: bool = True) -> nn.Module:
    """
    Tạo model theo model_type, load pretrained ImageNet weights.

    Args:
        model_type (str): "efficientnet_b7" hoặc "swin_transformer"
        pretrained (bool): Có load pretrained weights không (mặc định True)

    Returns:
        nn.Module: FundusRegressor với 1 output neuron

    Raises:
        ValueError: Nếu model_type không có trong REGISTRY
    """
    if model_type not in REGISTRY:
        raise ValueError(
            f"model_type '{model_type}' không hợp lệ. "
            f"Chọn một trong: {list(REGISTRY.keys())}"
        )

    timm_name = REGISTRY[model_type]

    # timm tự tải pretrained weights từ HuggingFace Hub nếu pretrained=True
    backbone = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=1,          # 1 neuron cho ordinal regression
    )

    model = FundusRegressor(backbone)
    return model


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_labels(raw_output: torch.Tensor) -> torch.Tensor:
    """
    Chuyển output thực của model thành nhãn nguyên [0, 4].
    Chấp nhận cả tensor shape [B, 1] lẫn [B] (đã squeeze trước đó).

    Args:
        raw_output (torch.Tensor): shape [B, 1] hoặc [B]
    Returns:
        torch.Tensor: shape [B], dtype long, giá trị trong [0, 4]
    """
    preds = raw_output.view(-1)             # flatten an toàn: [B,1] hoặc [B] → [B]
    preds = torch.round(preds).long()       # làm tròn
    preds = preds.clamp(min=0, max=4)       # clip về [0, 4]
    return preds
