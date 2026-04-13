import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7 as tv_efficientnet_b7, EfficientNet_B7_Weights

_SWIN_TIMM_NAME = "swinv2_base_window12to24_192to384_22kft1k"


_FEATURE_DIM = {
    "efficientnet_b7": 2560,
    "swinv2_base_384": 1024,
}


class FundusRegressor(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        head_type: str,
        feat_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        dropout_in: float | None = None,
        dropout_out: float | None = None,
    ):
        super().__init__()
        self.backbone  = backbone
        self.head_type = head_type
        d_in  = dropout if dropout_in is None else dropout_in
        d_out = dropout if dropout_out is None else dropout_out


        if head_type == "regression":
            self.head = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Dropout(d_in),
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(d_out),
                nn.Linear(hidden_dim, 1),
            )
        elif head_type == "ordinal":
            self.head = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Dropout(d_in),
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(d_out),
                nn.Linear(hidden_dim, 4),  
            )
        else:
            raise ValueError(
                f"head_type '{head_type}' không hợp lệ. Dùng 'regression' hoặc 'ordinal'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   
        return self.head(features)    


class SwinRegressor(nn.Module):


    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone  = backbone
        self.head_type = "regression"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)   # [B, 1]


def apply_freeze_strategy(model: nn.Module, model_type: str, strategy: str) -> nn.Module:
    if strategy == "none":
        return model

    # Bước 1: Đóng băng toàn bộ model
    for param in model.parameters():
        param.requires_grad = False

    # Bước 2: Unfeeze cho head
    # EfficientNet dùng custom self.head | SwinV2 dùng backbone.get_classifier()
    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        for param in model.backbone.get_classifier().parameters():
            param.requires_grad = True

    # Bước 3: unfreeze một phần backbone 
    if strategy == "partial":
        if "efficientnet_b7" in model_type:
            for name, param in model.backbone.named_parameters():
                if "features.7" in name or "features.8" in name:
                    param.requires_grad = True

        elif "swinv2_base_384" in model_type:
            # Swin Transformer có 4 stages (layers.0 đến layers.3). Ta unfreeze stage cuối (layers.3) và norm.
            for name, param in model.backbone.named_parameters():
                if "layers.3" in name or "norm" in name:
                    param.requires_grad = True

    return model


_VALID_MODEL_TYPES = {"efficientnet_b7", "swinv2_base_384"}


def build_model(
    model_type: str,
    head_type: str = "regression",
    pretrained: bool = True,
    freeze_strategy: str = "none",
    head_hidden_dim: int | None = None,
    head_dropout: float | None = None,
    head_dropout_in: float | None = None,
    head_dropout_out: float | None = None,
) -> nn.Module:
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type '{model_type}' không hợp lệ. Chọn: {_VALID_MODEL_TYPES}")

    if model_type == "efficientnet_b7":
        if head_type not in ("regression", "ordinal"):
            raise ValueError(f"head_type '{head_type}' không hợp lệ. Dùng 'regression' hoặc 'ordinal'.")

        # torchvision EfficientNet-B7 — tải từ PyTorch CDN
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_efficientnet_b7(weights=weights)
        backbone.classifier = nn.Identity()  # output [B, 2560] feature vector

        feat_dim = _FEATURE_DIM["efficientnet_b7"]   # 2560
        h_dim = head_hidden_dim if head_hidden_dim is not None else 512
        drop  = head_dropout if head_dropout is not None else 0.5
        model = FundusRegressor(
            backbone,
            head_type=head_type,
            feat_dim=feat_dim,
            hidden_dim=h_dim,
            dropout=drop,
            dropout_in=head_dropout_in,
            dropout_out=head_dropout_out,
        )

    else: 
        if head_type != "regression":
            print(
                f"[build_model] Cảnh báo: head_type='{head_type}' không được hỗ trợ "
                f"cho '{model_type}'. Tự động dùng regression."
            )
        import timm  #  chỉ cần khi dùng SwinV2
        backbone = timm.create_model(_SWIN_TIMM_NAME, pretrained=pretrained, num_classes=1)
        model = SwinRegressor(backbone)

    model = apply_freeze_strategy(model, model_type, freeze_strategy)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    effective_head   = head_type if model_type == "efficientnet_b7" else "regression (fixed)"
    print(
        f"[{model_type}] head={effective_head} | strategy={freeze_strategy} | "
        f"trainable: {trainable_params:,} / {total_params:,}"
    )

    return model


def predict_labels(
    raw_output: torch.Tensor,
    head_type: str = "regression",
    thresholds: list[float] | None = None,
) -> torch.Tensor:
    if head_type == "regression":
        preds = raw_output.view(-1).cpu().float()
        if thresholds is not None:
            import numpy as np
            bins     = np.sort(thresholds)
            preds_np = np.digitize(preds.numpy(), bins=bins).clip(0, 4)
            return torch.from_numpy(preds_np).long()
        return torch.round(preds.clamp(0.0, 4.0)).long()

    # ordinal
    preds = (torch.sigmoid(raw_output) > 0.5).sum(dim=1).long()
    return preds.clamp(0, 4)


def ordinal_label_transform(labels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(4, device=labels.device)       
    return (labels.unsqueeze(1) > thresholds).float()         