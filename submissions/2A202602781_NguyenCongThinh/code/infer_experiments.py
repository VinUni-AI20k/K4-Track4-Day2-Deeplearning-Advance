"""infer_experiments.py - Bước 3 của GUIDE: so sánh các phương pháp suy luận trên VAL + đo độ trễ.

[Implemented by Claude (AI assistant)]
Không huấn luyện lại. Dùng mô hình seed 0 của công thức chung kết (decisions["final_source"]) và các mô hình
backbone B0x (cho ensemble). Mọi chỉ số ở đây tính trên VAL; test không được đụng tới.

Chạy từ thư mục bài nộp:  python code/infer_experiments.py   (thường gọi qua run_all.py inference)
Ghi ra:
    results/inference.csv        mỗi dòng một phương pháp (sheet Inference)
    results/latency.csv          mỗi dòng một điều kiện đo (sheet Latency)
    figures/inference_tradeoff.png  macro-F1 val theo độ trễ p50 batch 1
    figures/reliability_val.png     biểu đồ độ tin cậy trước/sau temperature scaling (val, 2-fold)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import dataset as ds_mod  # noqa: E402
import experiments as ex  # noqa: E402
import inference as inf  # noqa: E402
import train  # noqa: E402

TRANSFORMER_KEYS = ("vit", "deit", "swin")


def is_transformer(name: str) -> bool:
    """[Implemented by Claude (AI assistant)] True nếu backbone cần kích thước ảnh cố định (ViT/DeiT/Swin)."""
    return any(k in name for k in TRANSFORMER_KEYS)


def val_loader(cfg, img_size: int | None = None, crop: bool = True, split: str = "val"):
    """[Implemented by Claude (AI assistant)] DataLoader của VAL (hoặc TEST, chỉ final.py dùng) cho suy luận.

    Input : cfg (Config); img_size (int | None) - kích thước crop (mặc định cfg.img_size);
            crop (bool) - True: resize + center-crop như I00; False: chỉ resize về eval_resize(img_size)
            (để views_multicrop tự cắt 5 crop); split ("val" | "test", test CHỈ dùng ở Bước 4).
    Output: DataLoader (x, y, filenames) đúng thứ tự file CSV của tập.
    """
    size = img_size or cfg.img_size
    if crop:
        tf = ds_mod.build_transforms(False, size)
    else:
        tf = v2.Compose([v2.ToImage(), v2.Resize(ds_mod.eval_resize(size), antialias=True),
                         v2.ToDtype(torch.float32, scale=True), v2.Normalize(ds_mod.IMAGENET_MEAN, ds_mod.IMAGENET_STD)])
    _, val_df, test_df = ds_mod.load_split(cfg.labels_dir, cfg.fold)
    df = {"val": val_df, "test": test_df}[split]
    # bộ đệm ảnh chỉ dùng ở kích thước train (đã có sẵn); kích thước khác đọc JPEG gốc (val chỉ 3.5k ảnh)
    preload = ds_mod.eval_resize(size) if cfg.preload and size == cfg.img_size else None
    return ds_mod.make_loader(df, cfg.images_dir, tf, 64, train=False, num_workers=cfg.num_workers,
                              preload_size=preload, cache_dir=cfg.cache_dir)


def metrics_probs(y: np.ndarray, probs: np.ndarray) -> dict:
    """[Implemented by Claude (AI assistant)] Chỉ số eval.py từ xác suất. Output: dict của eval.compute_metrics."""
    return train.compute_metrics(y, probs.argmax(1), probs)


def cv_temperature_ece(logits: np.ndarray, y: np.ndarray, folds: int = 2, seed: int = 0) -> dict:
    """[Implemented by Claude (AI assistant)] ECE trước/sau temperature scaling trên VAL, đánh giá chéo.

    Input : logits [N, 9] (val), y [N], folds (int), seed.
    Output: dict {"ece_before": float, "ece_after_cv": float (khớp T trên nửa này, đo ECE trên nửa kia),
                  "ece_after_insample": float, "T_full": float (khớp trên toàn bộ val - T dùng cho test)}
    Cách làm: tránh báo ECE "sau" lạc quan do khớp và đo trên cùng ảnh.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    after = np.zeros_like(logits, dtype=np.float64)
    for k in range(folds):
        fit_idx = np.concatenate([parts[j] for j in range(folds) if j != k])
        T = inf.fit_temperature(logits[fit_idx], y[fit_idx])
        after[parts[k]] = inf.apply_temperature(logits[parts[k]], T)
    T_full = inf.fit_temperature(logits, y)
    before = inf.apply_temperature(logits, 1.0)
    return {"ece_before": train.compute_metrics(y, before.argmax(1), before)["ece"],
            "ece_after_cv": train.compute_metrics(y, after.argmax(1), after)["ece"],
            "ece_after_insample": train.compute_metrics(y, before.argmax(1), inf.apply_temperature(logits, T_full))["ece"],
            "T_full": T_full, "probs_before": before, "probs_after_cv": after}


def reliability_plot(y, probs_before, probs_after, path: Path, bins: int = 15) -> None:
    """[Implemented by Claude (AI assistant)] Biểu đồ độ tin cậy (accuracy theo độ tin cậy) trước/sau TS.

    Input : y [N], probs_before/probs_after [N, 9], path (.png), bins. Output: None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for probs, label in ((probs_before, "trước TS"), (probs_after, "sau TS (2-fold trên val)")):
        conf = probs.max(1)
        corr = probs.argmax(1) == y
        idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
        xs, ys = [], []
        for m in range(bins):
            if (idx == m).any():
                xs.append(conf[idx == m].mean()); ys.append(corr[idx == m].mean())
        ax.plot(xs, ys, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="hiệu chuẩn hoàn hảo")
    ax.set_xlabel("độ tin cậy (max softmax)"); ax.set_ylabel("accuracy"); ax.set_title("Reliability diagram (val)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


def lat(model, img_size, device, k=1, batch=1, dtype="fp32", iters=100):
    """[Implemented by Claude (AI assistant)] Độ trễ (ms) K lượt forward ở batch cho trước, đo đúng cách.

    Input : model, img_size, device ("cuda"/"cpu"), k (số view/lượt), batch, dtype, iters.
    Output: dict của benchmark.latency_report (k=1) hoặc benchmark.tta_latency (k>1).
    """
    if k == 1:
        return benchmark.latency_report(model, batch, img_size, dtype=dtype, device=device, warmup=10, iters=iters)
    return benchmark.tta_latency(model, k, batch_size=batch, img_size=img_size, dtype=dtype, device=device,
                                 warmup=10, iters=iters)


def main() -> None:
    """[Implemented by Claude (AI assistant)] Chạy toàn bộ Bước 3 và ghi kết quả + lựa chọn suy luận.

    Input : không (đọc results/decisions.json). Output: None; ghi results/inference.csv, results/latency.csv,
            hình trong figures/, và decisions["inference"] = phương pháp cho F01 (chọn trên val).
    Luật chọn phương pháp cho F01: trong các phương pháp áp dụng được cho TỪNG mô hình (1-view, TTA lật,
      multi-crop, độ phân giải), lấy macro-F1 val cao nhất; nếu không hơn I00 ít nhất 0.002 thì giữ I00 (rẻ hơn).
      Temperature scaling (T khớp trên val) luôn áp dụng sau cùng vì không đổi accuracy.
    """
    dec = ex.load_decisions()
    src = dec["final_source"]
    cfg = ex.get_cfg(src)
    dev = train.resolve_device(cfg.device)
    device = dev.type
    model = train.load_trained_model(cfg, dev)
    S = cfg.img_size
    rows, lat_rows = [], []
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"CPU: {benchmark.cpu_name()}"

    def add(exp_id, method, k, probs, latency, note="", model_desc=f"{src} seed0 ({cfg.backbone})", ece=None):
        m = metrics_probs(y, probs)
        rows.append({"exp_id": exp_id, "method": method, "model": model_desc, "K": k,
                     "macro_f1_val": m["macro_f1"], "top1_val": m["top1"],
                     "ece_val": m["ece"] if ece is None else ece,
                     "f1_chinee_val": m["f1"][0], "f1_snake_val": m["f1"][7],
                     "p50_ms": latency.get("p50"), "p95_ms": latency.get("p95"), "p99_ms": latency.get("p99"),
                     "img_per_s_b32": latency.get("b32_img_s"), "note": note})
        print(f"[inference] {exp_id} {method}: F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} "
              f"p50 {latency.get('p50', float('nan')):.2f} ms", flush=True)

    def latpack(mdl, size, k=1, dtype="fp32"):
        b1 = lat(mdl, size, device, k=k, batch=1, dtype=dtype)
        b32 = lat(mdl, size, device, k=k, batch=32, dtype=dtype, iters=50)
        for r, b in ((b1, 1), (b32, 32)):
            lat_rows.append({"config": f"{cfg.backbone} {size}px K={k}", "gpu": gpu_name, "dtype": dtype,
                             "batch": b, "bn_fused": False, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
                             "img_per_s": r["images_per_s"], "torch": torch.__version__})
        return {"p50": b1["p50"], "p95": b1["p95"], "p99": b1["p99"], "b32_img_s": b32["images_per_s"]}

    # ---- I00 / I01 / I03: 1 view, lật ngang, gộp prob vs logit (một lượt đọc dữ liệu) ----
    names, y, (z0, zf) = inf.predict_logits_multiview(model, val_loader(cfg), dev, lambda x: [x, inf.view_hflip(x)])
    L1 = latpack(model, S)
    L2 = latpack(model, S, k=2)
    add("I00", "1 view (resize + center crop)", 1, inf.aggregate_views([z0], "prob"), L1)
    add("I01", "TTA lật ngang, gộp xác suất", 2, inf.aggregate_views([z0, zf], "prob"), L2)
    add("I03", "TTA lật ngang, gộp logit", 2, inf.aggregate_views([z0, zf], "logit"), L2)

    # ---- I02: 5 crop và 10 crop (5 crop + lật) ----
    _, _, crops = inf.predict_logits_multiview(model, val_loader(cfg, crop=False), dev,
                                               lambda x: inf.views_multicrop(x, S, flip=True))
    L5 = latpack(model, S, k=5)
    L10 = latpack(model, S, k=10)
    add("I02", "TTA 5 crop, gộp xác suất", 5, inf.aggregate_views(crops[:5], "prob"), L5)
    add("I02b", "TTA 10 crop (5 crop + lật), gộp xác suất", 10, inf.aggregate_views(crops, "prob"), L10)
    add("I03b", "TTA 10 crop, gộp logit", 10, inf.aggregate_views(crops, "logit"), L10)

    # ---- I04: dò độ phân giải kiểm tra (FixRes) - chỉ CNN ----
    res_probs = {}
    if not is_transformer(cfg.backbone):
        for s in (S - 32, S + 32, S + 64, S + 96):
            _, _, (zs,) = inf.predict_logits_multiview(model, val_loader(cfg, img_size=s), dev, lambda x: [x])
            res_probs[s] = inf.aggregate_views([zs], "prob")
            add(f"I04_{s}", f"1 view ở độ phân giải {s} (train {S})", 1, res_probs[s], latpack(model, s),
                note=f"resize {ds_mod.eval_resize(s)} + center crop {s}")
    else:
        rows.append({"exp_id": "I04", "method": "dò độ phân giải", "note":
                     f"không áp dụng: {cfg.backbone} có positional embedding/cửa sổ cố định {S} px"})

    # ---- I05: ensemble (trung bình xác suất) với 2 backbone B tốt nhất khác (val_logits đã lưu) ----
    b_runs = []
    for k in ex.BACKBONES:
        c = ex.backbone_cfg(k)
        f = train.run_dir(c) / "val_logits.npy"
        if f.exists() and c.backbone != cfg.backbone:
            s = json.loads((train.run_dir(c) / "summary.json").read_text(encoding="utf-8"))
            b_runs.append((s["val_macro_f1"], k, c))
    b_runs.sort(reverse=True)
    members = b_runs[:2]
    if members:
        probs = [inf.aggregate_views([z0], "prob")]
        lat_sum = dict(L1)
        for _, k, c in members:
            assert (train.run_dir(c) / "val_filenames.txt").read_text(encoding="utf-8").split("\n") == list(names)
            probs.append(train.softmax_np(np.load(train.run_dir(c) / "val_logits.npy")))
            Lm = latpack(train.load_trained_model(c, dev), c.img_size)
            for key in ("p50", "p95", "p99"):
                lat_sum[key] += Lm[key]
            lat_sum["b32_img_s"] = 1.0 / sum(1.0 / v for v in (lat_sum["b32_img_s"], Lm["b32_img_s"]))
        add("I05", "Ensemble 3 mô hình (trung bình xác suất)", 1 + len(members), inf.ensemble_probs(probs), lat_sum,
            model_desc=f"{src} + " + " + ".join(k for _, k, _ in members),
            note="độ trễ = tổng các mô hình chạy tuần tự")

    # ---- I06: trọng số EMA so với raw (cùng epoch) từ run T11 ----
    try:
        c11 = ex.get_cfg("T11")
        r11 = train.run_dir(c11)
        if (r11 / "val_logits_raw.npy").exists():
            add("I06_raw", "T11: trọng số raw (không EMA)", 1, train.softmax_np(np.load(r11 / "val_logits_raw.npy")), L1,
                model_desc="T11 seed0", note="cùng epoch với checkpoint EMA")
            add("I06", "T11: trọng số EMA (miễn phí lúc suy luận)", 1, train.softmax_np(np.load(r11 / "val_logits.npy")),
                L1, model_desc="T11 seed0")
    except RuntimeError:
        pass

    # ---- I07: temperature scaling + ECE (T khớp trên VAL; báo ECE "sau" bằng 2-fold trên val) ----
    ts = cv_temperature_ece(z0.astype(np.float64), y)
    add("I07", f"Temperature scaling (T={ts['T_full']:.3f} khớp trên val)", 1, ts["probs_after_cv"], L1,
        ece=ts["ece_after_cv"], note=f"ECE trước {ts['ece_before']:.4f} -> sau {ts['ece_after_cv']:.4f} (2-fold); "
                                      f"in-sample {ts['ece_after_insample']:.4f}")
    reliability_plot(y, ts["probs_before"], ts["probs_after_cv"], Path("figures/reliability_val.png"))

    # ---- I08: gộp BatchNorm (FP32) và AMP/FP16 ----
    fused = inf.fuse_conv_bn(model)
    err = inf.check_fusion(model, fused, S, dev)
    if fused.n_fused_bn:
        _, _, (zfu,) = inf.predict_logits_multiview(fused, val_loader(cfg), dev, lambda x: [x])
        Lfu = latpack(fused, S)
        lat_rows[-1]["bn_fused"] = lat_rows[-2]["bn_fused"] = True
        add("I08", f"Gộp BN vào conv ({fused.n_fused_bn} cặp), FP32", 1, inf.aggregate_views([zfu], "prob"), Lfu,
            note=f"sai số logit lớn nhất {err:.2e}")
    else:
        rows.append({"exp_id": "I08", "method": "Gộp BN", "note": f"không áp dụng: {cfg.backbone} không có BatchNorm"})
    if dev.type == "cuda":
        _, _, (zamp,) = inf.predict_logits_multiview(model, val_loader(cfg), dev, lambda x: [x], amp=True)
        add("I08b", "AMP (autocast FP16)", 1, inf.aggregate_views([zamp], "prob"), latpack(model, S, dtype="amp"))
    # độ trễ trên CPU (giống máy tính nhúng của robot), batch 1, FP32 - chỉ để tham khảo triển khai
    cpu_model = train.load_trained_model(cfg, "cpu")
    rc = benchmark.latency_report(cpu_model, 1, S, dtype="fp32", device="cpu", warmup=10, iters=50)
    lat_rows.append({"config": f"{cfg.backbone} {S}px K=1 (CPU)", "gpu": f"CPU: {benchmark.cpu_name()}",
                     "dtype": "fp32", "batch": 1, "bn_fused": False, "p50": rc["p50"], "p95": rc["p95"],
                     "p99": rc["p99"], "img_per_s": rc["images_per_s"], "torch": torch.__version__})
    # (bonus) ONNX Runtime trên CPU so với PyTorch CPU
    try:
        onnx_row = onnx_latency(cpu_model, S)
        lat_rows.append({**onnx_row, "config": f"{cfg.backbone} {S}px ONNX Runtime (CPU)"})
    except Exception as e:  # onnx không bắt buộc
        print(f"[inference] bỏ qua ONNX: {e}")

    out = Path("results"); out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    base_p50 = df.loc[df["exp_id"] == "I00", "p50_ms"].iloc[0]
    df["rel_cost_vs_I00"] = df["p50_ms"] / base_p50
    df.to_csv(out / "inference.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)
    tradeoff_plot(df, Path("figures/inference_tradeoff.png"))

    # ---- chọn phương pháp cho F01 (chỉ trên VAL) ----
    cand = {"I00": ("none", "prob", S), "I01": ("hflip", "prob", S), "I03": ("hflip", "logit", S),
            "I02": ("5crop", "prob", S), "I02b": ("10crop", "prob", S), "I03b": ("10crop", "logit", S)}
    cand.update({f"I04_{s}": ("none", "prob", s) for s in res_probs})
    d = df[df["exp_id"].isin(cand)].set_index("exp_id")
    best = d["macro_f1_val"].idxmax()
    if d.loc[best, "macro_f1_val"] - d.loc["I00", "macro_f1_val"] < 0.002:
        best = "I00"
    method, space, size = cand[best]
    ex.save_decision(inference={"exp_id": best, "method": method, "space": space, "img_size": int(size),
                                "temperature_scaling": True},
                     reasons={"inference": f"{best} có macro-F1 val {d.loc[best, 'macro_f1_val']:.4f} "
                                           f"(I00 {d.loc['I00', 'macro_f1_val']:.4f}); luật: cao nhất trên val, "
                                           f"phải hơn I00 >= 0.002, sau đó temperature scaling khớp trên val"})
    print(df.to_string())


def onnx_latency(model, img_size: int) -> dict:
    """[Implemented by Claude (AI assistant)] Xuất ONNX và đo độ trễ ONNX Runtime CPU, batch 1 (bonus RUBRIC).

    Input : model (trên CPU, eval), img_size. Output: dict một dòng cho sheet Latency.
    """
    import tempfile

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "m.onnx")
        torch.onnx.export(model.eval(), torch.randn(1, 3, img_size, img_size), path, input_names=["x"],
                          output_names=["y"], opset_version=17, dynamo=False)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        r = benchmark.bench(lambda: sess.run(None, {"x": x}), warmup=10, iters=50)
    return {"gpu": f"CPU: {benchmark.cpu_name()}", "dtype": "fp32", "batch": 1, "bn_fused": "onnx",
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "img_per_s": 1000.0 / r["p50"],
            "torch": f"onnxruntime {ort.__version__}"}


def tradeoff_plot(df: pd.DataFrame, path: Path) -> None:
    """[Implemented by Claude (AI assistant)] Scatter macro-F1 val theo độ trễ p50 batch 1 (đường đánh đổi).

    Input : df (bảng inference), path. Output: None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["p50_ms", "macro_f1_val"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(d["p50_ms"], d["macro_f1_val"])
    for _, r in d.iterrows():
        ax.annotate(r["exp_id"], (r["p50_ms"], r["macro_f1_val"]), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("độ trễ p50 batch 1 (ms, thang log)"); ax.set_ylabel("macro-F1 val")
    ax.set_title("Đánh đổi độ chính xác - độ trễ của các phương pháp suy luận"); ax.grid(alpha=0.3)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
