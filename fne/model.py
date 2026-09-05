import torchvision
import torchvision.models as models
import torch.nn as nn

TORCHVISION_VERSION = torchvision.__version__
CONVNEXT_AVAILABLE = False
try:
    version_parts = [int(part) for part in TORCHVISION_VERSION.split(".")[:2]]
    if version_parts[0] > 0 or (version_parts[0] == 0 and version_parts[1] >= 13):
        CONVNEXT_AVAILABLE = True
except Exception:
    pass

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        h_out = self.h_conv(self.h_avg_pool(x)).expand(-1, -1, -1, w)
        w_out = self.w_conv(self.w_avg_pool(x)).expand(-1, -1, h, -1)
        return x * h_out * w_out


def _replace_relu_with_silu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU(inplace=True))
        else:
            _replace_relu_with_silu(child)


def _build_tv_model(ctor, pretrained):
    try:
        return ctor(pretrained=pretrained)
    except TypeError:
        return ctor(weights="DEFAULT" if pretrained else None)


def _named_ctors(names_dims):
    mapping = {}
    for name, dim in names_dims:
        ctor = getattr(models, name, None)
        if ctor is not None:
            mapping[name] = (ctor, dim)
    return mapping


class ImprovedGazeModel(nn.Module):
    def __init__(self, backbone="convnext_small", num_outputs=1, pretrained=True):
        super().__init__()
        self.backbone_type = backbone

        if backbone.startswith("vit"):
            if not TIMM_AVAILABLE:
                raise ImportError("timm库未安装，请运行: pip install timm")
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0
            )
            feature_dim = self.backbone.embed_dim
            self.vit_to_cam = nn.Sequential(
                nn.Linear(feature_dim, 768),
                nn.SiLU(inplace=True),
                nn.Unflatten(1, (768, 1, 1)),
                nn.ConvTranspose2d(768, feature_dim, kernel_size=7, stride=1, padding=0),
                nn.SiLU(inplace=True),
            )
            self.coordinate_attn = CoordinateAttention(feature_dim)

        elif backbone.startswith("resnet"):
            resnet_map = _named_ctors([
                ("resnet18", 512),
                ("resnet34", 512),
                ("resnet50", 2048),
                ("resnet101", 2048),
            ])
            if backbone not in resnet_map:
                raise ValueError(f"不支持的ResNet backbone: {backbone}")
            ctor, feature_dim = resnet_map[backbone]
            resnet_model = _build_tv_model(ctor, pretrained)
            _replace_relu_with_silu(resnet_model)
            self.backbone = nn.Sequential(*list(resnet_model.children())[:-2])
            self.coordinate_attn = CoordinateAttention(feature_dim)

        elif backbone.startswith("convnext"):
            if not CONVNEXT_AVAILABLE:
                raise ImportError(
                    f"ConvNeXt需要torchvision 0.13.0或更高版本。当前版本: {TORCHVISION_VERSION}"
                )
            convnext_map = _named_ctors([
                ("convnext_tiny", 768),
                ("convnext_small", 768),
                ("convnext_base", 1024),
                ("convnext_large", 1536),
                ("convnext_xlarge", 2048),
            ])
            if backbone not in convnext_map:
                raise ValueError(f"不支持的ConvNeXt backbone: {backbone}")
            ctor, feature_dim = convnext_map[backbone]
            convnext_model = _build_tv_model(ctor, pretrained)
            self.backbone = nn.Sequential(*list(convnext_model.children())[:-2])
            self.coordinate_attn = CoordinateAttention(feature_dim)

        else:
            raise ValueError(f"不支持的backbone类型: {backbone}")

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        if self.backbone_type.startswith("vit"):
            features = self.vit_to_cam(features)
        features = self.coordinate_attn(features)
        features = nn.AdaptiveAvgPool2d((1, 16))(features)
        return self.regression_head(features)


def create_model(model_type="convnext_small", pretrained=True, num_outputs=1):
    if not (model_type.startswith("resnet")
            or model_type.startswith("vit")
            or model_type.startswith("convnext")):
        raise ValueError(f"不支持的模型类型: {model_type}")
    return ImprovedGazeModel(
        backbone=model_type, num_outputs=num_outputs, pretrained=pretrained
    )


def list_available_backbones():
    backbones = {
        "resnet": ["resnet18", "resnet34", "resnet50", "resnet101"],
        "convnext": [],
        "vit": [],
    }
    if CONVNEXT_AVAILABLE:
        backbones["convnext"] = [
            "convnext_tiny", "convnext_small", "convnext_base",
            "convnext_large", "convnext_xlarge",
        ]
    if TIMM_AVAILABLE:
        vit_models = [
            "vit_base_patch16_224",
            "vit_base_patch16_384",
            "vit_large_patch16_224",
            "vit_large_patch16_384",
            "vit_small_patch16_224",
            "vit_tiny_patch16_224",
        ]
        for model_name in vit_models:
            try:
                timm.create_model(model_name, pretrained=False, num_classes=0)
                backbones["vit"].append(model_name)
            except Exception:
                pass
    return backbones


if __name__ == "__main__":
    print("可用Backbone列表:")
    available = list_available_backbones()
    print(f"ResNet: {available['resnet']}")
    print(f"ViT: {available['vit']}")
    print(f"ConvNeXt: {available['convnext']}")
