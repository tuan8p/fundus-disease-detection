import torch
import torch.nn as nn
import timm

REGISTRY = {
    "efficientnet_b7": "tf_efficientnet_b7",
    "swinv2_base_384": "swinv2_base_window12to24_192to384_22kft1k"
}

class FundusRegressor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def apply_freeze_strategy(model: nn.Module, model_type: str, strategy: str) -> nn.Module:
    """
    Áp dụng chiến lược đóng băng trọng số.
    Các mode:
      - "none"      : Unfreeze all (Train toàn bộ network).
      - "head_only" : Freeze toàn bộ backbone, chỉ train classifier head (Linear Probing).
      - "partial"   : Freeze các layer đầu, chỉ unfreeze vài block/stage cuối và head.
    """
    if strategy == "none":
        return model

    # Bước 1: Đóng băng toàn bộ model
    for param in model.parameters():
        param.requires_grad = False

    # Bước 2: Luôn luôn mở băng (unfreeze) cho Classifier Head
    # timm hỗ trợ lấy head qua get_classifier()
    for param in model.backbone.get_classifier().parameters():
        param.requires_grad = True

    # Bước 3: Mở băng một phần tùy theo kiến trúc (nếu là "partial")
    if strategy == "partial":
        if "efficientnet" in model_type:
            # EfficientNet có các blocks (thường từ 0 đến 6). Ta unfreeze block cuối (blocks.6) và conv_head.
            for name, param in model.backbone.named_parameters():
                if "blocks.6" in name or "conv_head" in name or "bn2" in name:
                    param.requires_grad = True
                    
        elif "swin" in model_type:
            # Swin Transformer có 4 stages (layers.0 đến layers.3). Ta unfreeze stage cuối (layers.3) và norm.
            for name, param in model.backbone.named_parameters():
                if "layers.3" in name or "norm" in name:
                    param.requires_grad = True

    return model

def build_model(model_type: str, pretrained: bool = True, freeze_strategy: str = "none") -> nn.Module:
    if model_type not in REGISTRY:
        raise ValueError(f"model_type '{model_type}' không hợp lệ.")

    timm_name = REGISTRY[model_type]
    backbone = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=1, 
    )
    
    model = FundusRegressor(backbone)
    
    # Áp dụng chiến lược freeze
    model = apply_freeze_strategy(model, model_type, freeze_strategy)
    
    # In ra số lượng tham số có requires_grad = True để kiểm tra
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_type}] Strategy: {freeze_strategy} | Trainable params: {trainable_params:,} / {total_params:,}")
    
    return model

def predict_labels(raw_output: torch.Tensor) -> torch.Tensor:
    preds = raw_output.view(-1)             
    preds = torch.round(preds).long()       
    preds = preds.clamp(min=0, max=4)       
    return preds