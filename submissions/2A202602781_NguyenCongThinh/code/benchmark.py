"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

# [Implemented by Claude (AI assistant)] imports for the implementation below.
import platform
import time

import numpy as np
import torch


def cpu_name() -> str:
    """[Implemented by Claude (AI assistant)] Tên CPU đọc từ registry Windows (fallback platform.processor()).

    Input : không. Output: str, ví dụ "AMD Ryzen 5 4500U with Radeon Graphics".
    """
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : fn (callable không tham số, một lượt suy luận); warmup (int, số lần chạy bỏ đi);
    #         iters (int, số lần đo, nên >= 50); sync (callable | None, ví dụ torch.cuda.synchronize; None trên CPU)
    # Output: dict {"p50", "p95", "p99", "mean", "std", "min", "max" (đơn vị ms), "n": iters, "warmup": warmup}
    # Cách làm: chạy warmup lần rồi bỏ; mỗi lần đo: sync() -> perf_counter -> fn() -> sync() -> perf_counter.
    #   Trên CPU PyTorch chạy đồng bộ nên sync=None vẫn đúng; trên GPU bắt buộc truyền synchronize.
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(times.mean()),
            "std": float(times.std(ddof=1)) if iters > 1 else float("nan"),
            "min": float(times.min()), "max": float(times.max()), "n": int(iters), "warmup": int(warmup)}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    # [Implemented by Claude (AI assistant)]
    # Input : model; batch_size (int); img_size (int); dtype "fp32" | "amp" | "fp16"; device "cuda" | "cpu"
    #         (nếu yêu cầu "cuda" mà máy không có CUDA thì tự chuyển sang "cpu" và ghi rõ); warmup; iters
    # Output: dict {"gpu": tên GPU hoặc "CPU: <tên CPU>", "device", "dtype", "batch", "img_size",
    #         "p50", "p95", "p99", "mean", "std" (ms), "images_per_s", "torch", "threads", "iters", "warmup"}
    # Cách làm: model.eval(); đầu vào ngẫu nhiên cố định (KHÔNG tính tiền xử lý/giải mã ảnh, chỉ forward);
    #   torch.inference_mode(); fp16 = model.half() (chỉ hỗ trợ trên CUDA); amp = autocast (CUDA: float16,
    #   CPU: bfloat16); sync = torch.cuda.synchronize trên GPU, None trên CPU (tính toán CPU là đồng bộ).
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    if dtype == "fp16" and dev.type != "cuda":
        raise ValueError("fp16 thuần chỉ đo trên CUDA; trên CPU hãy dùng fp32 hoặc amp (bfloat16)")
    model = model.eval().to(dev)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model = model.half()
        x = x.half()
    amp_dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=dtype == "amp"):
            model(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else f"CPU: {cpu_name()}"
    return {"gpu": name, "device": dev.type, "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"], "std": r["std"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "iters": iters, "warmup": warmup}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    # [Implemented by Claude (AI assistant)]
    # Input : model; k_views (int, số view); kw: batch_size (mặc định 1), img_size (mặc định 224),
    #         dtype ("fp32"), device ("cuda" -> tự về cpu nếu không có), warmup (10), iters (100)
    # Output: dict {"k_views", "p50", "p95", "p99", "mean" (ms, cho cả K view), "single_p50" (ms, 1 view),
    #         "k_times_single_p50" (= K * single_p50), "ratio_vs_single" (= p50 / single_p50), "images_per_s", ...}
    # Cách làm: đo thật hàm chạy K lượt forward liên tiếp (giống TTA tuần tự từng view, batch 1) và đo riêng
    #   một lượt, để so với giả định "chi phí ~ K lần".
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    device = kw.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = kw.get("dtype", "fp32")
    warmup, iters = kw.get("warmup", 10), kw.get("iters", 100)
    dev = torch.device(device)
    model = model.eval().to(dev)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    amp_dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16

    def fn_k():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=dtype == "amp"):
            for _ in range(k_views):
                model(x)

    single = latency_report(model, batch_size, img_size, dtype=dtype, device=device, warmup=warmup, iters=iters)
    r = bench(fn_k, warmup=warmup, iters=iters, sync=sync)
    return {**{k: single[k] for k in ("gpu", "device", "dtype", "batch", "img_size", "torch", "threads")},
            "k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
            "single_p50": single["p50"], "k_times_single_p50": k_views * single["p50"],
            "ratio_vs_single": r["p50"] / single["p50"], "images_per_s": batch_size / (r["p50"] / 1000.0)}
