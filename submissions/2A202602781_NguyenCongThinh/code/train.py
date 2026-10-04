"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.

# [Implemented by Claude (AI assistant)] imports for the implementation below.
import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _find_repo_root() -> Path:
    """[Implemented by Claude (AI assistant)] Tìm thư mục chứa eval.py (repo gốc) bằng cách đi ngược lên.

    Input : không. Output: Path thư mục chứa eval.py. Dùng để `from eval import ...` mà không sửa eval.py.
    """
    if os.environ.get("LAB_REPO_ROOT"):  # cho phép chạy bản sao của code ở nơi khác (ví dụ kiểm tra nhanh)
        return Path(os.environ["LAB_REPO_ROOT"]).resolve()
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "eval.py").exists() and (p / "RUBRIC.md").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import compute_metrics, save_predictions  # noqa: E402  (eval.py gốc, không sửa)

import dataset as ds_mod  # noqa: E402
import losses as loss_mod  # noqa: E402
import model as model_mod  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- [Implemented by Claude (AI assistant)] các trường thêm vào (README cho phép thêm tham số) ---
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png (rỗng = tên backbone)
    optimizer: str = "adamw"          # adamw | sgd  (trục E)
    momentum: float = 0.9             # chỉ dùng cho sgd
    drop_path_rate: float = 0.0       # stochastic depth (trục F)
    grad_clip: float | None = None    # clip chuẩn gradient, None = tắt
    device: str = "auto"              # auto | cpu | cuda
    num_threads: int | None = None    # số luồng CPU cho torch (None = mặc định của torch)
    channels_last: bool = False       # bố trí bộ nhớ NHWC (nhanh hơn với conv trên GPU/oneDNN)
    preload: bool = True              # đọc ảnh từ bộ đệm uint8 .npy thay vì giải mã JPEG mỗi epoch
    cache_dir: str = "runs/cache"     # nơi đặt bộ đệm ảnh (bị .gitignore)
    curves_dir: str = "curves"        # nơi lưu ảnh biểu đồ training
    limit_train_batches: int | None = None  # chỉ dùng cho kiểm tra nhanh (smoke test), None = cả epoch
    resume: bool = True               # tiếp tục từ runs/<exp_id>/seed<k>/last.pt nếu phiên bị ngắt


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : seed (int). Output: None.
    # Cách làm: cố định random, numpy, torch CPU và mọi GPU; PYTHONHASHSEED. Seed của worker DataLoader
    #   được suy ra từ generator của loader (dataset.make_loader nhận cùng seed + dataset.seed_worker).
    #   cudnn.benchmark=False, deterministic=True để tái lập tốt hơn trên GPU (chậm hơn một chút).
    #   Trên CPU (máy dùng cho bài nộp này) cùng seed cho kết quả lặp lại gần như y hệt.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    # [Implemented by Claude (AI assistant)]
    # Input : model, cfg (Config: optimizer, lr_backbone, lr_head, weight_decay, momentum)
    # Output: torch.optim.Optimizer (AdamW mặc định, hoặc SGD Nesterov momentum cho trục E)
    # Cách làm: chia tham số bằng model.param_groups (backbone decay / backbone no-decay / head),
    #   mỗi nhóm có lr và weight_decay riêng; scheduler sau đó nhân hệ số chung cho mọi nhóm.
    groups = model_mod.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr_backbone)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr_backbone, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer!r}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    # [Implemented by Claude (AI assistant)]  -> cập nhật THEO BƯỚC (mỗi iteration).
    # Input : optimizer; cfg (epochs, warmup_epochs); steps_per_epoch (int, số batch train mỗi epoch)
    # Output: torch.optim.lr_scheduler.LambdaLR; hệ số f(step) nhân với lr gốc của TỪNG nhóm
    #   (nên tỉ lệ LR head/backbone = 10 được giữ suốt quá trình).
    # Cách làm: warmup tuyến tính f = (step+1)/W trong W = warmup_epochs * steps_per_epoch bước đầu,
    #   sau đó cosine f = 0.5 * (1 + cos(pi * t)) với t đi từ 0 tới 1 ở bước cuối (LR về 0).
    total = max(1, int(cfg.epochs * steps_per_epoch))
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        # [Implemented by Claude (AI assistant)]
        # Input : model (nn.Module đang train), decay (float, ví dụ 0.998)
        # Output: đối tượng EMA; self.module là BẢN SAO RIÊNG của model ở chế độ eval, không gradient
        #   (đánh giá bằng self.module, không ghi đè model đang train).
        self.decay = float(decay)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        # [Implemented by Claude (AI assistant)]
        # Input : model (cùng kiến trúc với self.module), gọi SAU mỗi optimizer.step(). Output: None.
        # Cách làm: d = min(decay, (1 + n) / (10 + n)) (warmup của decay như timm, để EMA không bị kéo
        #   bởi trọng số khởi tạo khi số bước ít) rồi W_ema <- d * W_ema + (1 - d) * W cho mọi tham số
        #   VÀ mọi buffer số thực (running_mean/var của BN cũng được trung bình động như timm ModelEmaV2);
        #   buffer số nguyên (num_batches_tracked) được chép thẳng.
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
                else:
                    v.copy_(src)

    def state_dict(self) -> dict:
        # [Implemented by Claude (AI assistant)] Output: state_dict của mô hình EMA (để lưu checkpoint).
        return self.module.state_dict()


def set_train_mode(model) -> None:
    """[Implemented by Claude (AI assistant)] model.train() nhưng giữ backbone đóng băng ở eval.

    Input : model (có thể có cờ model.frozen_backbone do model.freeze_backbone đặt). Output: None.
    Cách làm: bật train cho toàn bộ; nếu frozen_backbone thì đặt lại eval() cho mọi module con KHÔNG thuộc
      head, để BatchNorm dùng/giữ nguyên running stats của ImageNet và dropout trong backbone tắt
      (GUIDE mục 3.2). Module gốc vẫn training=True để dropout trước head (drop_rate) còn hoạt động.
    """
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = model_mod.head_module_names(model)
        for name, m in model.named_modules():
            if name and name not in head:
                m.training = False  # không đệ quy, để các module con thuộc head vẫn ở train


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model, loader (train DataLoader -> (x, y, filenames)), criterion, optimizer,
    #         scheduler (LambdaLR theo bước), scaler (torch.amp.GradScaler), cfg, device, ema (EMA | None)
    # Output: dict {"train_loss": float (trung bình theo mẫu), "train_acc": float (NaN khi Mixup/CutMix vì
    #         không có nghĩa), "lr": float (LR nhóm đầu ở cuối epoch), "lr_steps": list[float] (LR nhóm đầu
    #         từng bước, để vẽ warmup + cosine), "time_s": float, "n_steps": int}
    # Cách làm: set_train_mode (giữ BN backbone đóng băng ở eval) -> mỗi batch: (mix_batch nếu cfg.mix) ->
    #   autocast (chỉ bật trên CUDA khi cfg.amp) -> loss (mixed_loss khi trộn) -> scaler.scale(loss).backward()
    #   -> clip gradient (tuỳ chọn) -> scaler.step -> scaler.update -> scheduler.step() -> ema.update.
    set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    t0 = time.perf_counter()
    tot_loss, tot_correct, tot_n, steps = 0.0, 0, 0, 0
    lr_steps = []
    for i, (x, y, _) in enumerate(loader):
        if cfg.limit_train_batches is not None and i >= cfg.limit_train_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = None
        if cfg.mix:
            x, targets = loss_mod.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_mod.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_steps.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        if ema is not None:
            ema.update(model)
        b = y.size(0)
        tot_loss += float(loss.detach()) * b
        tot_n += b
        if targets is None:
            tot_correct += int((logits.detach().argmax(1) == y).sum())
        steps += 1
    return {
        "train_loss": tot_loss / max(1, tot_n),
        "train_acc": (tot_correct / max(1, tot_n)) if not cfg.mix else float("nan"),
        "lr": lr_steps[-1] if lr_steps else float("nan"),
        "lr_steps": lr_steps,
        "time_s": time.perf_counter() - t0,
        "n_steps": steps,
    }


def evaluate(model, loader, criterion, device, channels_last: bool = False):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model, loader (val/test DataLoader, shuffle=False), criterion (loss dùng để báo cáo val loss;
    #         train.run truyền CE thường để đường val loss so sánh được giữa các loss), device,
    #         channels_last (bool, tham số thêm) - đổi bố trí bộ nhớ đầu vào giống lúc train
    # Output: (filenames list[str] [N], y_true ndarray int64 [N], logits ndarray float32 [N, 9], loss float)
    # Cách làm: model.eval() + torch.inference_mode(); duyệt loader theo đúng thứ tự, gom logit trên CPU.
    model.eval()
    names, ys, outs = [], [], []
    tot_loss, tot_n = 0.0, 0
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            logits = model(x).float()
            tot_loss += float(criterion(logits, y)) * y.size(0)
            tot_n += y.size(0)
            names.extend(f)
            ys.append(y.cpu().numpy())
            outs.append(logits.cpu().numpy())
    return names, np.concatenate(ys).astype(np.int64), np.concatenate(outs).astype(np.float32), tot_loss / max(1, tot_n)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """[Implemented by Claude (AI assistant)] Softmax ổn định số học trên numpy.

    Input : logits ndarray [N, K]. Output: ndarray float64 [N, K], mỗi dòng cộng bằng 1.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """[Implemented by Claude (AI assistant)] Chỉ số theo ĐÚNG định nghĩa của eval.py.

    Input : y_true [N] int, logits [N, 9]. Output: dict của eval.compute_metrics
      {"n", "top1", "macro_f1", "balanced_acc", "ece", "nll", "precision"[9], "recall"[9], "f1"[9],
       "support"[9], "confusion"[9, 9]}.
    """
    probs = softmax_np(logits)
    return compute_metrics(np.asarray(y_true), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps: list[float] | None = None) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : history - list[dict], mỗi phần tử là một epoch: {"epoch", "train_loss", "train_acc", "val_loss",
    #           "val_macro_f1", "val_top1", (tuỳ chọn) "val_macro_f1_raw", ...}
    #         path (đường dẫn .png), title (str, ví dụ "B01 - resnet18"),
    #         lr_steps (list[float] | None, tham số thêm) - LR nhóm backbone theo từng bước
    # Output: None; ghi file .png 3 ô: (1) loss train/val, (2) macro-F1 val, top-1 val, acc train,
    #         (3) LR theo bước (warmup + cosine). Epoch tốt nhất (macro-F1 val) được đánh dấu.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    n_panels = 3 if lr_steps else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2))
    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], "o-", label="train loss")
    ax.plot(h["epoch"], h["val_loss"], "s-", label="val loss (CE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(h["epoch"], h["val_macro_f1"], "o-", label="val macro-F1")
    ax.plot(h["epoch"], h["val_top1"], "s--", label="val top-1")
    if "val_macro_f1_raw" in h and h["val_macro_f1_raw"].notna().any():
        ax.plot(h["epoch"], h["val_macro_f1_raw"], "^:", label="val macro-F1 (raw, no EMA)")
    if h["train_acc"].notna().any():
        ax.plot(h["epoch"], h["train_acc"], "x-.", label="train acc (augmented)")
    best = int(h["val_macro_f1"].idxmax())
    ax.axvline(h["epoch"][best], color="gray", lw=0.8, ls=":")
    ax.annotate(f"best ep {int(h['epoch'][best])}: {h['val_macro_f1'][best]:.4f}",
                (h["epoch"][best], h["val_macro_f1"][best]), textcoords="offset points", xytext=(-60, -18))
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_title("Validation metrics")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")

    if lr_steps:
        ax = axes[2]
        ax.plot(np.arange(len(lr_steps)), lr_steps)
        ax.set_xlabel("step"); ax.set_ylabel("LR (backbone group)"); ax.set_title("LR schedule")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def resolve_device(name: str) -> torch.device:
    """[Implemented by Claude (AI assistant)] "auto" -> cuda nếu có, ngược lại cpu. Output: torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def env_info(device: torch.device) -> dict:
    """[Implemented by Claude (AI assistant)] Thông tin môi trường để ghi vào config.json (tái lập).

    Output: dict {"python", "torch", "torchvision", "timm", "device", "device_name", "num_threads", "platform"}.
    """
    import timm
    import torchvision

    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(device)
    else:
        dev_name = platform.processor() or platform.machine()
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "device": device.type, "device_name": dev_name,
            "num_threads": torch.get_num_threads(), "platform": platform.platform()}


def curve_path(cfg: Config) -> Path:
    """[Implemented by Claude (AI assistant)] curves/<exp_id>_<desc>.png (seed 0) hoặc ..._seed<k>.png.

    Input : cfg. Output: Path ảnh biểu đồ; tên bắt đầu bằng exp_id để khớp results.xlsx (GUIDE 6.1).
    """
    desc = cfg.desc or cfg.backbone
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    # [Implemented by Claude (AI assistant)]
    # Input : cfg (Config) - toàn bộ cấu hình một lần chạy.
    # Output: dict tóm tắt (cũng ghi ra runs/<exp_id>/seed<k>/summary.json):
    #   {"exp_id", "seed", "backbone", "weight_tag", "init", "img_size", "epochs", "best_epoch",
    #    "val_macro_f1", "val_top1", "val_balanced_acc", "val_ece", "val_nll",
    #    "val_f1_per_class": [9 số], "val_recall_per_class": [9 số], "params_m", "gmacs",
    #    "train_time_per_epoch_s", "val_time_per_epoch_s", "total_time_s", "device", "curve", "config": {...}}
    # File ghi ra trong run_dir: config.json, split_check.json, history.csv, best.pt (trọng số tốt nhất;
    #   với EMA là trọng số EMA + bản raw), val_logits.npy, val_pred.csv, summary.json (+ test_logits.npy khi
    #   save_test_predictions). Ảnh curves/<exp_id>_<desc>.png.
    # Luồng: set_seed -> load_split + check_split -> loaders -> build_model/criterion/optimizer/scheduler/EMA
    #   -> (resume từ last.pt nếu có) -> mỗi epoch: train_one_epoch -> evaluate(val) -> chọn best theo macro-F1 val
    #   (bằng nhau giữ epoch sớm hơn: so sánh ">" chặt) -> nạp best -> lưu logit/predictions val
    #   -> (chỉ khi save_test_predictions) đánh giá test ĐÚNG MỘT LẦN -> history.csv, biểu đồ, summary.
    #   Nếu lần chạy với cùng cấu hình đã xong (summary.json tồn tại) thì không train lại; khi đó nếu được yêu
    #   cầu test mà chưa có, chỉ nạp best.pt và chạy test một lần (dùng ở Bước 4 cho mốc T00 seed 0).
    t_start = time.perf_counter()
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)

    summary_file = rd / "summary.json"
    if cfg.resume and summary_file.exists():
        prev = json.loads(summary_file.read_text(encoding="utf-8"))
        if _cfg_key(prev["config"]) == _cfg_key(dataclasses.asdict(cfg)):
            print(f"[run] {cfg.exp_id} seed{cfg.seed}: đã xong trước đó, dùng lại {summary_file}")
            if cfg.save_test_predictions and not (rd / "test_logits.npy").exists():
                _predict_and_save(cfg, "val", device)
                _predict_and_save(cfg, "test", device)
            return prev
        raise RuntimeError(f"{rd} đã có kết quả của cấu hình KHÁC; đổi exp_id/out_dir hoặc xoá thư mục đó")

    # 1. seed + config
    set_seed(cfg.seed)
    (rd / "config.json").write_text(json.dumps({"config": dataclasses.asdict(cfg), "env": env_info(device)},
                                               indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. split + kiểm tra bắt buộc
    train_df, val_df, test_df = ds_mod.load_split(cfg.labels_dir, cfg.fold)
    split_report = ds_mod.check_split(train_df, val_df, test_df, cfg.images_dir)
    (rd / "split_check.json").write_text(json.dumps(split_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. loaders (test KHÔNG được tạo ở đây)
    train_loader = _make_split_loader(cfg, train_df, train=True)
    val_loader = _make_split_loader(cfg, val_df, train=False)

    # 4. model, loss, optimizer, scheduler, scaler, EMA
    model = model_mod.build_model(cfg.backbone, pretrained=True, num_classes=ds_mod.NUM_CLASSES,
                                  drop_rate=cfg.drop_rate, init=cfg.init, img_size=cfg.img_size,
                                  drop_path_rate=cfg.drop_path_rate)
    params_m = model_mod.count_params(model)
    gmacs = model_mod.count_gmacs(model, cfg.img_size)
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    crit_kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=ds_mod.NUM_CLASSES)
        crit_kw["weight"] = loss_mod.class_weights(counts, beta=cfg.class_weight_beta or 0.0)
    criterion = loss_mod.build_criterion(cfg.loss, **crit_kw).to(device)
    val_criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader) if cfg.limit_train_batches is None \
        else min(len(train_loader), cfg.limit_train_batches)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history, lr_all = [], []
    best_f1, best_epoch, start_epoch = -1.0, -1, 1
    last_file = rd / "last.pt"
    if cfg.resume and last_file.exists():
        ck = torch.load(last_file, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        if ema is not None:
            ema.module.load_state_dict(ck["ema"]); ema.num_updates = ck["ema_updates"]
        history, lr_all = ck["history"], ck["lr_all"]
        best_f1, best_epoch, start_epoch = ck["best_f1"], ck["best_epoch"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng_torch"]); np.random.set_state(ck["rng_numpy"]); random.setstate(ck["rng_py"])
        train_loader.generator.set_state(ck["rng_loader"])
        print(f"[run] tiếp tục {cfg.exp_id} seed{cfg.seed} từ epoch {start_epoch}")

    # 5. vòng epoch
    for epoch in range(start_epoch, cfg.epochs + 1):
        stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_val = time.perf_counter()
        eval_model = ema.module if ema is not None else model
        _, yv, lv, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.channels_last)
        m = metrics_from_logits(yv, lv)
        row = {"epoch": epoch, "train_loss": stats["train_loss"], "train_acc": stats["train_acc"],
               "val_loss": vloss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"], "lr": stats["lr"],
               "train_time_s": stats["time_s"]}
        if ema is not None:  # so sánh trọng số raw và EMA (dùng cho I06)
            _, yr, lr_, _ = evaluate(model, val_loader, val_criterion, device, cfg.channels_last)
            mr = metrics_from_logits(yr, lr_)
            row.update(val_macro_f1_raw=mr["macro_f1"], val_top1_raw=mr["top1"])
        row["val_time_s"] = time.perf_counter() - t_val
        history.append(row)
        lr_all.extend(stats["lr_steps"])

        if m["macro_f1"] > best_f1:  # ">" chặt: hoà thì giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save({"model": eval_model.state_dict(),
                        "raw_model": model.state_dict() if ema is not None else None,
                        "epoch": epoch, "val_macro_f1": best_f1, "config": dataclasses.asdict(cfg)},
                       rd / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.module.state_dict() if ema is not None else None,
                    "ema_updates": ema.num_updates if ema is not None else 0,
                    "history": history, "lr_all": lr_all, "best_f1": best_f1, "best_epoch": best_epoch,
                    "epoch": epoch, "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
                    "rng_py": random.getstate(), "rng_loader": train_loader.generator.get_state()}, last_file)
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch}/{cfg.epochs} loss {stats['train_loss']:.4f} "
              f"val_loss {vloss:.4f} val_F1 {m['macro_f1']:.4f} val_top1 {m['top1']:.4f} "
              f"({stats['time_s']:.0f}s train, {row['val_time_s']:.0f}s val)", flush=True)

    # 6. nạp best, lưu logit + dự đoán val
    names_v, yv, lv = _predict_and_save(cfg, "val", device)
    mv = metrics_from_logits(yv, lv)
    best_ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    if best_ck.get("raw_model") is not None:  # logit val của trọng số raw cùng epoch (cho I06)
        raw = load_trained_model(cfg, device, which="raw")
        _, _, lraw, _ = evaluate(raw, val_loader, val_criterion, device, cfg.channels_last)
        np.save(rd / "val_logits_raw.npy", lraw)

    # 7. test: CHỈ ở Bước 4, đúng một lần
    if cfg.save_test_predictions:
        _predict_and_save(cfg, "test", device)

    # 8. history, biểu đồ, tóm tắt
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    cpath = curve_path(cfg)
    plot_curves(history, cpath, f"{cfg.exp_id} seed{cfg.seed} - {cfg.backbone} ({cfg.desc or 'baseline'})", lr_all)
    hist = pd.DataFrame(history)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": model.weight_tag,
        "init": cfg.init, "img_size": cfg.img_size, "epochs": cfg.epochs, "best_epoch": best_epoch,
        "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_balanced_acc": mv["balanced_acc"],
        "val_ece": mv["ece"], "val_nll": mv["nll"],
        "val_f1_per_class": mv["f1"].tolist(), "val_recall_per_class": mv["recall"].tolist(),
        "params_m": params_m, "gmacs": gmacs,
        "train_time_per_epoch_s": float(hist["train_time_s"].mean()),
        "val_time_per_epoch_s": float(hist["val_time_s"].mean()),
        "total_time_s": float(hist["train_time_s"].sum() + hist["val_time_s"].sum()),
        "wall_time_this_session_s": time.perf_counter() - t_start,
        "device": env_info(device)["device_name"], "curve": str(cpath),
        "config": dataclasses.asdict(cfg),
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if last_file.exists():
        last_file.unlink()  # đã xong, bỏ trạng thái tối ưu để tiết kiệm đĩa (best.pt vẫn giữ)
    return summary


# các trường không ảnh hưởng tới kết quả huấn luyện; bỏ qua khi so cấu hình để dùng lại một lần chạy đã xong
_VOLATILE_KEYS = ("save_test_predictions", "num_workers", "resume", "num_threads", "pred_dir", "curves_dir",
                  "out_dir", "images_dir", "labels_dir", "cache_dir", "device")


def _cfg_key(d: dict) -> dict:
    """[Implemented by Claude (AI assistant)] Bỏ các trường "không ảnh hưởng kết quả" khỏi dict cấu hình.

    Input : dict cấu hình (dataclasses.asdict). Output: dict đã lọc để so sánh hai lần chạy có cùng thiết lập.
    """
    return {k: v for k, v in d.items() if k not in _VOLATILE_KEYS}


def _make_split_loader(cfg: Config, df: pd.DataFrame, train: bool, img_size: int | None = None):
    """[Implemented by Claude (AI assistant)] DataLoader cho một tập theo cfg.

    Input : cfg; df (DataFrame của tập); train (bool); img_size (int | None) - ghi đè cfg.img_size
            (dùng cho dò độ phân giải kiểm tra). Output: DataLoader (xem dataset.make_loader).
    Cách làm: transform train hoặc eval từ dataset.build_transforms; bộ đệm ảnh kích thước eval_resize(img_size).
    """
    size = img_size or cfg.img_size
    tf = ds_mod.build_transforms(train, size, cfg.aug)
    preload = ds_mod.eval_resize(size) if cfg.preload else None
    return ds_mod.make_loader(df, cfg.images_dir, tf, cfg.batch_size if train else cfg.batch_size * 2,
                              train=train, sampler=cfg.sampler if train else None,
                              num_workers=cfg.num_workers, preload_size=preload, cache_dir=cfg.cache_dir,
                              seed=cfg.seed)


def load_trained_model(cfg: Config, device: torch.device | str = "cpu", which: str = "best"):
    """[Implemented by Claude (AI assistant)] Dựng lại model và nạp trọng số từ runs/<exp_id>/seed<k>/best.pt.

    Input : cfg (Config của lần chạy), device, which ("best" = trọng số dùng để chọn checkpoint, tức EMA nếu
            có EMA; "raw" = trọng số không EMA ở cùng epoch, chỉ có khi train với EMA)
    Output: nn.Module ở chế độ eval trên device. Được dùng bởi run() (bước 6-7), infer_experiments.py, final.py.
    """
    ck = torch.load(run_dir(cfg) / "best.pt", map_location="cpu", weights_only=False)
    m = model_mod.build_model(cfg.backbone, pretrained=False, num_classes=ds_mod.NUM_CLASSES,
                              drop_rate=cfg.drop_rate, init="finetune", img_size=cfg.img_size,
                              drop_path_rate=cfg.drop_path_rate)
    state = ck["model"] if which == "best" else ck["raw_model"]
    if state is None:
        raise ValueError(f"checkpoint {run_dir(cfg)} không có trọng số '{which}'")
    m.load_state_dict(state)
    m.to(device)
    if cfg.channels_last:
        m.to(memory_format=torch.channels_last)
    return m.eval()


def _predict_and_save(cfg: Config, split: str, device) -> tuple[list[str], np.ndarray, np.ndarray]:
    """[Implemented by Claude (AI assistant)] Dự đoán 1-view (I00) một tập bằng best.pt và lưu kết quả.

    Input : cfg; split ("val" | "test"); device.
    Output: (filenames list[str], y_true ndarray [N], logits ndarray [N, 9]).
    Ghi: run_dir/<split>_logits.npy, run_dir/<split>_pred.csv (định dạng eval.py), và nếu
      cfg.save_test_predictions: predictions/<exp_id>_seed<k>_<split>.csv (eval.save_predictions).
    Test chỉ được gọi khi cfg.save_test_predictions=True (assert), đúng quy tắc S4.
    """
    if split == "test":
        assert cfg.save_test_predictions, "test chỉ chạy ở Bước 4 (save_test_predictions=True)"
    train_df, val_df, test_df = ds_mod.load_split(cfg.labels_dir, cfg.fold)
    df = {"val": val_df, "test": test_df}[split]
    model = load_trained_model(cfg, device)
    loader = _make_split_loader(cfg, df, train=False)
    names, y, logits, _ = evaluate(model, loader, nn.CrossEntropyLoss(), device, cfg.channels_last)
    rd = run_dir(cfg)
    np.save(rd / f"{split}_logits.npy", logits)
    (rd / f"{split}_filenames.txt").write_text("\n".join(names), encoding="utf-8")
    save_predictions(rd / f"{split}_pred.csv", names, y, softmax_np(logits))
    if cfg.save_test_predictions:
        save_predictions(pred_path(cfg, split), names, y, softmax_np(logits))
    return names, y, logits


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : pairs (list[str]) dạng "KEY=VALUE"
    # Output: dict {tên field: giá trị đã ép kiểu}, ví dụ {"seed": 1, "loss": "focal", "ema_decay": None}
    # Cách làm: đọc kiểu khai báo của từng field trong Config (chuỗi như "int", "float | None", "str | None",
    #   "bool" vì file dùng `from __future__ import annotations`); "none"/"null" -> None nếu kiểu cho phép;
    #   bool nhận true/false/1/0/yes/no; sau đó int/float/str. Key lạ -> ValueError liệt kê các key hợp lệ.
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        key, val = pair.split("=", 1)
        key = key.strip()
        if key not in types:
            raise ValueError(f"không có field '{key}' trong Config; hợp lệ: {sorted(types)}")
        t = types[key]
        raw = val.strip()
        if raw.lower() in ("none", "null") and "None" in t:
            out[key] = None
        elif t.startswith("bool"):
            if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key}: '{raw}' không phải bool")
            out[key] = raw.lower() in ("true", "1", "yes")
        elif t.startswith("int"):
            out[key] = int(raw)
        elif t.startswith("float"):
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : dòng lệnh `--set KEY=VALUE ...`. Output: None; in JSON tóm tắt (bỏ khối config cho gọn).
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình Lab Day 2")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè field của Config")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    res = run(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
