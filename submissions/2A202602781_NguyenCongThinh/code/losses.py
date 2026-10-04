"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm/lớp có `raise NotImplementedError`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

# [Implemented by Claude (AI assistant)] imports for the implementation below.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    TODO: tạo đúng loss, hoặc gọi các lớp bên dưới.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : kind (str) - "ce" | "ls" | "focal" | "ce_weighted"
    #         kw: smoothing (float, cho "ls", mặc định 0.1); gamma (float, cho "focal", mặc định 2.0);
    #             alpha (tensor[9] | None, cho "focal"); weight (tensor[9], BẮT BUỘC cho "ce_weighted")
    # Output: nn.Module gọi được dạng criterion(logits[B, 9], target[B] int64) -> tensor vô hướng (trung bình batch)
    # Cách làm: chọn lớp loss tương ứng. "ce_weighted" dùng nn.CrossEntropyLoss(weight=...) với trọng số
    #   tính từ số ảnh TRAIN bằng class_weights(); trung bình có trọng số theo định nghĩa của PyTorch.
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=tensor[9] (dùng class_weights trên số ảnh train)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind!r}")


class LabelSmoothingCE(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    TODO: tự cài đặt hoặc dùng torch.nn.CrossEntropyLoss(label_smoothing=eps), rồi ghi rõ
    bạn đã chọn cách nào. Kiểm tra: eps = 0 phải cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        # [Implemented by Claude (AI assistant)] TỰ CÀI ĐẶT (không dùng tham số label_smoothing của PyTorch).
        # Input : smoothing (float) eps trong [0, 1). Output: module loss.
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        # [Implemented by Claude (AI assistant)]
        # Input : logits (tensor [B, K] float), target (tensor [B] int64)
        # Output: tensor vô hướng = mean_b sum_k -q'(k) log p(k)
        # Cách làm: -sum_k q'(k) log p(k) = (1 - eps) * (-log p_y) + eps * mean_k(-log p_k);
        #   eps = 0 cho lại đúng CE (kiểm tra trong test_codes.py).
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    TODO:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        # [Implemented by Claude (AI assistant)]
        # Input : gamma (float >= 0); alpha (None | sequence/tensor độ dài K) trọng số theo lớp.
        # Output: module loss. alpha được lưu dạng buffer để tự chuyển theo .to(device).
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        # [Implemented by Claude (AI assistant)]
        # Input : logits [B, K], target [B] int64. Output: tensor vô hướng (trung bình batch).
        # Cách làm: log p_t = log_softmax(logits)[y]; p_t = exp(log p_t);
        #   loss_b = -(1 - p_t)^gamma * log p_t (* alpha[y] nếu có), rồi lấy mean.
        #   gamma = 0, alpha = None => đúng bằng cross-entropy (kiểm tra trong test_codes.py).
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt).clamp(min=0.0) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    TODO: trả về tensor độ dài 9. Chỉ dùng số liệu của train, không dùng val hay test.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : counts (sequence/ndarray độ dài 9) - số ảnh mỗi lớp trong TẬP TRAIN; beta (float, 0 <= beta < 1)
    # Output: torch.FloatTensor[9]; trung bình = 1 (tức tổng = 9) ở cả hai chế độ.
    # Cách làm: beta=0: w_c = 1/n_c; beta>0: w_c = (1 - beta) / (1 - beta^n_c) (Cui et al.).
    #   Sau đó chuẩn hoá w <- w * K / sum(w). train.run gọi hàm này với số đếm của train_df.
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mỗi lớp phải có ít nhất 1 ảnh train")
    if beta == 0:
        w = 1.0 / n
    else:
        if not 0.0 < beta < 1.0:
            raise ValueError("beta phải trong [0, 1)")
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    w = w * len(n) / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    TODO: tự cài đặt. Kiểm tra bằng mắt: vẽ vài ảnh sau khi trộn và in lam.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : x (tensor [B, C, H, W]), y (tensor [B] int64), alpha (float > 0) tham số Beta,
    #         mode (str) "mixup" | "cutmix"
    # Output: (x_mix [B, C, H, W], (y_a [B], y_b [B], lam float)) với y_a = y, y_b = y[perm]
    # Cách làm: lam ~ Beta(alpha, alpha) (numpy, đã được train.set_seed cố định); perm = hoán vị batch
    #   (torch.randperm). mixup: nội suy tuyến tính hai ảnh. cutmix: hộp có cạnh H*sqrt(1-lam) x W*sqrt(1-lam),
    #   tâm ngẫu nhiên, bị cắt theo biên ảnh; dán vùng đó từ x[perm] vào bản sao của x, rồi tính LẠI
    #   lam = 1 - diện tích hộp thật / (H*W) để nhãn khớp đúng tỉ lệ pixel. Được gọi bởi train.train_one_epoch.
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        H, W = x.shape[-2:]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
        x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(H * W)
    else:
        raise ValueError(f"mode không hợp lệ: {mode!r}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    TODO. Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : criterion (callable loss), logits [B, K], targets = (y_a, y_b, lam) từ mix_batch
    # Output: tensor vô hướng = lam * L(logits, y_a) + (1 - lam) * L(logits, y_b)
    #   (với CE, biểu thức này tương đương CE với nhãn mềm lam*onehot(y_a) + (1-lam)*onehot(y_b)).
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
