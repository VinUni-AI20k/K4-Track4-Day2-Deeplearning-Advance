"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# [Implemented by Claude (AI assistant)] imports for the implementation below.
import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}

# [Implemented by Claude (AI assistant)] backbone nhỏ dùng cho ngân sách CPU của bài nộp này
# (máy không có GPU CUDA; xem report.md mục 2). Vẫn đủ 4 ràng buộc của GUIDE mục 2.1.
CPU_BACKBONES = {
    "resnet18": "resnet18",                     # họ ResNet (mốc)
    "convnext_atto": "convnext_atto",           # họ ConvNeXt
    "deit_tiny": "deit_tiny_patch16_224",       # transformer (DeiT)
    "mobilenetv3": "mobilenetv3_large_100",     # mạng nhẹ
    "efficientnet_b0": "efficientnet_b0",       # mạng nhẹ
}
# họ kiến trúc cần biết kích thước ảnh lúc tạo model (positional embedding / cửa sổ attention)
_NEEDS_IMG_SIZE = ("vit", "deit", "swin", "eva", "beit")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", img_size: int | None = None,
                drop_path_rate: float = 0.0, head_init: str = "zero"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    # [Implemented by Claude (AI assistant)]
    # Input : name (str) - tên model trong timm (ví dụ "resnet18", "deit_tiny_patch16_224")
    #         pretrained (bool) - tải trọng số ImageNet; bị ép False khi init="scratch"
    #         num_classes (int) - số lớp của head mới (9); drop_rate (float) - dropout trước head
    #         init (str) - "scratch" | "frozen" | "finetune" (trục A)
    #         img_size (int | None) - chỉ truyền cho ViT/DeiT/Swin (nội suy positional embedding)
    #         drop_path_rate (float) - stochastic depth (trục F), 0 = tắt
    #         head_init (str) - "zero": đặt trọng số + bias của head mới = 0 để logit ban đầu bằng 0 và loss ban đầu
    #             đúng bằng ln 9 = 2.197 (GUIDE 1.3). Lý do: với head khởi tạo mặc định của timm, MobileNetV3 có loss
    #             ban đầu ~4.7 (đặc trưng 1280 chiều sau conv_head có độ lớn cao) - kiểm tra pipeline phát hiện ra.
    #             "default": giữ khởi tạo của timm (chỉ dùng để so sánh trong eda.pipeline_checks).
    # Output: nn.Module của timm, có thêm 2 thuộc tính:
    #         model.weight_tag (str)  - tag trọng số thật sự tải, ví dụ "resnet18.a1_in1k" (hoặc "<name> (scratch)")
    #         model.frozen_backbone (bool) - True nếu đã đóng băng (train.set_train_mode đọc cờ này)
    # Cách làm: timm.create_model thay head bằng Linear(num_features, 9) khởi tạo ngẫu nhiên; với
    #   init="frozen" gọi freeze_backbone. Được gọi bởi train.run, inference/final script khi nạp checkpoint.
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init!r}")
    if init == "scratch":
        pretrained = False
    kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    if img_size is not None and any(k in name for k in _NEEDS_IMG_SIZE):
        kw["img_size"] = img_size
    model = timm.create_model(name, **kw)

    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", name)
    tag = cfg.get("tag")
    model.weight_tag = (f"{arch}.{tag}" if tag else arch) if pretrained else f"{name} (scratch)"
    if head_init == "zero":
        head = model.get_classifier()
        nn.init.zeros_(head.weight)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
    elif head_init != "default":
        raise ValueError(f"head_init không hợp lệ: {head_init!r}")
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model (nn.Module timm có get_classifier())
    # Output: None (sửa tại chỗ): mọi tham số ngoài head có requires_grad=False; model.frozen_backbone=True
    # Cách làm: lấy id các tham số của head (model.get_classifier()), tắt grad phần còn lại.
    #   BN: model.train() ở đầu mỗi epoch sẽ bật lại train mode cho BN, nên train.set_train_mode()
    #   (gọi trong train_one_epoch) đặt lại mọi module ngoài head về eval() khi cờ frozen_backbone bật,
    #   để running_mean/var của backbone không bị cập nhật.
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def head_module_names(model) -> set[str]:
    """[Implemented by Claude (AI assistant)] Tên (theo named_modules) của head và các module con của nó.

    Input : model (timm nn.Module). Output: set[str] tên module thuộc head (dùng để giữ backbone ở eval).
    """
    head = model.get_classifier()
    return {n for n, m in model.named_modules() if m is head or any(m is c for c in head.modules())}


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model; lr_backbone (float); lr_head (float); weight_decay (float)
    # Output: list[dict], mỗi dict là một nhóm cho torch.optim:
    #   [{"name": "backbone_decay",    "params": [...], "lr": lr_backbone, "weight_decay": weight_decay},
    #    {"name": "backbone_no_decay", "params": [...], "lr": lr_backbone, "weight_decay": 0.0},
    #    {"name": "head",              "params": [...], "lr": lr_head,     "weight_decay": weight_decay},
    #    {"name": "head_no_decay",     "params": [...], "lr": lr_head,     "weight_decay": 0.0}]
    #   (nhóm rỗng bị bỏ, ví dụ init="frozen" chỉ còn 2 nhóm head).
    # Cách làm: tham số thuộc model.get_classifier() là head; còn lại là backbone. ndim <= 1
    #   (bias, gamma/beta của BN/LayerNorm, layer-scale, ...) không weight decay. Đây là 3 nhóm của slide
    #   trang 52; riêng bias của head cũng được tách ra để quy tắc "không decay norm/bias" (GUIDE 1.4)
    #   áp dụng nhất quán. Được gọi bởi train.build_optimizer.
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    buckets = {"backbone_decay": [], "backbone_no_decay": [], "head": [], "head_no_decay": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        no_decay = p.ndim <= 1
        key = ("head" if is_head else "backbone") + ("_no_decay" if no_decay else ("" if is_head else "_decay"))
        buckets[key].append(p)
    settings = {
        "backbone_decay": (lr_backbone, weight_decay),
        "backbone_no_decay": (lr_backbone, 0.0),
        "head": (lr_head, weight_decay),
        "head_no_decay": (lr_head, 0.0),
    }
    return [{"name": k, "params": v, "lr": settings[k][0], "weight_decay": settings[k][1]}
            for k, v in buckets.items() if v]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng. TODO."""
    # [Implemented by Claude (AI assistant)]
    # Input : model. Output: float - tổng số phần tử của mọi tham số / 1e6 (kể cả tham số đóng băng).
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO: dùng thư viện đếm (fvcore, ptflops, thop...) hoặc tự đếm bằng hook.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model; img_size (int) - cạnh ảnh vuông đầu vào
    # Output: float - số GMAC (tỉ phép nhân-cộng) cho MỘT ảnh 3 x img_size x img_size
    # Cách làm: chạy một lượt forward với torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch,
    #   đếm conv/linear/matmul/attention; KHÔNG đếm BN, activation, pooling). FLOPs của nó = 2 x MAC,
    #   nên GMAC = FLOPs / 2 / 1e9. Trạng thái train/eval của model được khôi phục sau khi đếm.
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9
