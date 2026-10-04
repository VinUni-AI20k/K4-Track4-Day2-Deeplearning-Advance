"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

# [Implemented by Claude (AI assistant)] imports for the implementation below.
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model; loader (DataLoader eval, shuffle=False, trả (x, y, filenames)); device;
    #         view (callable | None) - hàm x[B,C,H,W] -> x' (MỘT batch), ví dụ view_hflip
    # Output: (filenames list[str] [N], y_true ndarray int64 [N], logits ndarray float32 [N, 9])
    # Cách làm: gọi predict_logits_multiview với một view duy nhất rồi lấy phần tử đầu.
    names, y, outs = predict_logits_multiview(model, loader, device, lambda x: [(view or view_identity)(x)])
    return names, y, outs[0]


def predict_logits_multiview(model, loader, device, views_fn, amp: bool = False):
    """[Implemented by Claude (AI assistant)] Một lượt duyệt loader, chạy model trên K view của mỗi batch.

    Input : model; loader; device; views_fn (callable x -> list[K batch]) ví dụ
            lambda x: [x, view_hflip(x)] hoặc lambda x: views_multicrop(x, 128);
            amp (bool) - autocast FP16 (chỉ có tác dụng trên CUDA)
    Output: (filenames list[str], y_true ndarray [N], list K phần tử ndarray float32 [N, 9] - logit từng view)
    Cách làm: model.eval() + torch.inference_mode(); giữ đúng thứ tự file; đọc dữ liệu một lần cho cả K view.
    """
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            views = views_fn(x)
            if outs is None:
                outs = [[] for _ in views]
            for k, v in enumerate(views):
                with torch.autocast(device_type=torch.device(device).type, enabled=amp and torch.device(device).type == "cuda"):
                    outs[k].append(model(v).float().cpu().numpy())
            names.extend(f)
            ys.append(np.asarray(y))
    return names, np.concatenate(ys).astype(np.int64), [np.concatenate(o).astype(np.float32) for o in outs]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    # [Implemented by Claude (AI assistant)] Input: x [N, C, H, W]. Output: x lật theo chiều W (dim=-1).
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    # [Implemented by Claude (AI assistant)]
    # Input : x [N, C, H, W] với H, W >= crop (loader đưa ảnh đã resize nhưng CHƯA center-crop);
    #         crop (int); flip (bool) - thêm bản lật ngang của mỗi crop (10 view)
    # Output: list 5 (hoặc 10) tensor [N, C, crop, crop]: trên-trái, trên-phải, dưới-trái, dưới-phải, giữa
    H, W = x.shape[-2:]
    if H < crop or W < crop:
        raise ValueError(f"ảnh {H}x{W} nhỏ hơn crop {crop}")
    top, left = (H - crop) // 2, (W - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, W - crop:], x[..., H - crop:, :crop],
             x[..., H - crop:, W - crop:], x[..., top:top + crop, left:left + crop]]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : x [N, C, H, W] (đã chuẩn hoá); sizes (iterable int) các cạnh đích
    # Output: list tensor [N, C, s, s], mỗi phần tử ứng với một s trong sizes (s == H thì giữ nguyên x)
    # Cách làm: F.interpolate bilinear, antialias=True khi thu nhỏ. Giới hạn: chỉ dùng cho CNN có global
    #   pooling (ResNet/MobileNet/EfficientNet/ConvNeXt). DeiT tạo với img_size cố định sẽ báo lỗi kích thước
    #   positional embedding, nên các thí nghiệm đa tỉ lệ của bài nộp chỉ chạy trên CNN.
    out = []
    for s in sizes:
        if (x.shape[-2], x.shape[-1]) == (s, s):
            out.append(x)
        else:
            out.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                                     antialias=s < x.shape[-1]))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : logits_per_view (list K ndarray [N, 9] hoặc ndarray [K, N, 9]); space "prob" | "logit"
    # Output: ndarray float64 [N, 9], mỗi dòng cộng bằng 1
    # Cách làm: "prob" = mean_k softmax(z_k); "logit" = softmax(mean_k z_k). Bài nộp so sánh cả hai (I03).
    z = np.asarray(logits_per_view, dtype=np.float64)
    if z.ndim == 2:
        z = z[None]
    if space == "prob":
        p = _softmax(z, axis=-1).mean(axis=0)
    elif space == "logit":
        p = _softmax(z.mean(axis=0), axis=-1)
    else:
        raise ValueError(f"space không hợp lệ: {space!r}")
    return p / p.sum(axis=1, keepdims=True)


def _softmax(z, axis=-1):
    """[Implemented by Claude (AI assistant)] Softmax ổn định số học (numpy). Input: ndarray; Output: ndarray cùng dạng."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : list_of_probs (list M ndarray [N, 9] xác suất, CÙNG thứ tự file). Output: ndarray [N, 9]
    # Cách làm: kiểm tra cùng dạng, lấy trung bình cộng rồi chuẩn hoá lại mỗi dòng về tổng 1.
    arrs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError("các mô hình phải dự đoán trên cùng tập ảnh (cùng dạng)")
    p = np.mean(arrs, axis=0)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : val_logits (ndarray [N, 9]) - logit trên VAL (hoặc log của xác suất TTA đã gộp);
    #         val_labels (ndarray [N] int)
    # Output: float T > 0 cực tiểu NLL = mean -log softmax(z / T)[y]
    # Cách làm: (1) quét lưới thô 400 giá trị log T trong [log 0.05, log 20] để tránh cực tiểu cục bộ;
    #   (2) tinh chỉnh bằng LBFGS trên tham số log T (đảm bảo T > 0), float64.
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.int64)

    def nll(log_t):
        return F.cross_entropy(z / torch.exp(log_t), y)

    grid = torch.linspace(np.log(0.05), np.log(20.0), 400, dtype=torch.float64)
    with torch.no_grad():
        losses = torch.stack([nll(g) for g in grid])
    log_t = grid[int(losses.argmin())].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(log_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t.detach()))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    # [Implemented by Claude (AI assistant)]
    # Input : logits ndarray [N, 9]; T (float > 0). Output: ndarray float64 [N, 9] xác suất (argmax không đổi).
    if T <= 0:
        raise ValueError("T phải > 0")
    return _softmax(np.asarray(logits, dtype=np.float64) / T, axis=1)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model (nn.Module, ví dụ timm resnet18 / mobilenetv3 / efficientnet)
    # Output: fused_model - BẢN SAO đã gộp (model gốc không bị sửa) ở chế độ eval; số cặp đã gộp được ghi ở
    #   thuộc tính fused_model.n_fused_bn (int). Kiến trúc không có cặp Conv2d -> BatchNorm2d (DeiT, ConvNeXt
    #   dùng LayerNorm) thì n_fused_bn = 0 và mô hình không đổi.
    # Cách làm: duyệt mọi module cha; với hai module con ĐĂNG KÝ liền nhau (conv, bn) mà bn là BatchNorm2d
    #   và số kênh khớp: tính w', b' bằng công thức trong docstring (torch.nn.utils.fusion.fuse_conv_bn_weights),
    #   ghi vào bản sao của conv (giữ nguyên lớp, stride, padding, groups), thay bn bằng Identity.
    #   timm dùng BatchNormAct2d (BN + drop + activation trong một module): khi đó thay bằng
    #   Sequential(drop, act) để KHÔNG làm mất hàm kích hoạt. Thứ tự đăng ký của timm trùng thứ tự forward
    #   với các cặp này; kết quả luôn được kiểm tra bằng check_fusion() (sai số lớn nhất phải ~1e-5).
    from torch.nn.utils.fusion import fuse_conv_bn_weights

    fused = copy.deepcopy(model).eval()
    n_fused = 0
    for parent in list(fused.modules()):
        children = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(children, children[1:]):
            if not (isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d)):
                continue
            if m1.out_channels != m2.num_features or not m2.track_running_stats:
                continue
            conv = copy.deepcopy(m1)
            w, b = fuse_conv_bn_weights(m1.weight, m1.bias, m2.running_mean, m2.running_var,
                                        m2.eps, m2.weight, m2.bias)
            conv.weight, conv.bias = w, b
            setattr(parent, n1, conv)
            if hasattr(m2, "act") and hasattr(m2, "drop"):  # timm BatchNormAct2d
                setattr(parent, n2, nn.Sequential(m2.drop, m2.act))
            else:
                setattr(parent, n2, nn.Identity())
            n_fused += 1
    fused.n_fused_bn = n_fused
    return fused


def check_fusion(model, fused, img_size: int = 224, device="cpu") -> float:
    """[Implemented by Claude (AI assistant)] Sai số lớn nhất giữa đầu ra trước và sau khi gộp BN.

    Input : model (gốc), fused (sau fuse_conv_bn), img_size, device.
    Output: float max |logit_gốc - logit_gộp| trên một batch ngẫu nhiên 4 ảnh (kỳ vọng cỡ 1e-5 trở xuống).
    """
    torch.manual_seed(0)
    x = torch.randn(4, 3, img_size, img_size, device=device)
    with torch.inference_mode():
        a = model.eval()(x)
        b = fused.eval()(x)
    return float((a - b).abs().max())
