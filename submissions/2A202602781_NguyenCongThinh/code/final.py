"""final.py - Bước 4 của GUIDE: chạy TEST đúng MỘT lần cho mỗi seed, ghi predictions/ và chấm bằng eval.py.

[Implemented by Claude (AI assistant)]
Điều kiện trước: các run F01_seed{0,1,2} và T00_seed{0,1,2} đã huấn luyện xong (run_all.py final_train),
và results/decisions.json đã có "inference" (chọn trên VAL ở Bước 3).

Chạy từ thư mục bài nộp:  python code/final.py        (thường gọi qua run_all.py final_test)
Ghi ra:
    predictions/T00_seed<k>_test.csv, T00_seed<k>_val.csv      mốc: công thức nền + 1 view (I00), T = 1
    predictions/F01_seed<k>_test.csv, F01_seed<k>_val.csv      chung kết: suy luận đã chọn + temperature scaling
    predictions/F01_uncal_seed<k>_test.csv                      chung kết trước temperature scaling (cho I4a)
    results/final_temperatures.json, results/eval/*.md|csv|json  kết quả eval.py score/grade
    figures/confusion_test_F01.png, figures/errors_chinee_snake.png
Bảo vệ quy tắc "test một lần": nếu đã có results/eval/TEST_DONE.json thì dừng (trừ khi --force, và khi đó
báo cáo phải ghi rõ lý do chạy lại).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import dataset as ds_mod  # noqa: E402
import experiments as ex  # noqa: E402
import inference as inf  # noqa: E402
import train  # noqa: E402
from infer_experiments import val_loader  # noqa: E402
from train import save_predictions  # noqa: E402  (eval.save_predictions của repo gốc)

DONE = Path("results/eval/TEST_DONE.json")


def views_for(method: str, size: int):
    """[Implemented by Claude (AI assistant)] Hàm tạo view và kiểu loader cho một phương pháp suy luận.

    Input : method ("none" | "hflip" | "5crop" | "10crop"), size (int).
    Output: (views_fn: x -> list[batch], crop: bool - loader có center-crop hay không, k: số view)
    """
    if method == "none":
        return (lambda x: [x]), True, 1
    if method == "hflip":
        return (lambda x: [x, inf.view_hflip(x)]), True, 2
    if method in ("5crop", "10crop"):
        flip = method == "10crop"
        return (lambda x: inf.views_multicrop(x, size, flip=flip)), False, 10 if flip else 5
    raise ValueError(method)


def predict_split(cfg, split: str, method: str, space: str, size: int, device):
    """[Implemented by Claude (AI assistant)] Dự đoán một tập bằng best.pt của cfg với phương pháp suy luận cho trước.

    Input : cfg (Config của run), split ("val" | "test"), method, space ("prob" | "logit"), size (int), device.
    Output: (filenames list[str], y_true ndarray [N], z ndarray [N, 9]) với z là "logit tương đương" sau khi gộp
            view: log của xác suất gộp (space="prob") hoặc trung bình logit (space="logit"); softmax(z) = xác suất
            gộp, nên temperature scaling áp dụng được thống nhất: softmax(z / T).
    """
    model = train.load_trained_model(cfg, device)
    views_fn, crop, _ = views_for(method, size)
    names, y, outs = inf.predict_logits_multiview(model, val_loader(cfg, size, crop=crop, split=split), device, views_fn)
    if space == "logit":
        z = np.mean(outs, axis=0)
    else:
        z = np.log(np.clip(inf.aggregate_views(outs, "prob"), 1e-12, None))
    return names, y, z.astype(np.float64)


def run_eval(args: list[str], out_md: Path) -> str:
    """[Implemented by Claude (AI assistant)] Gọi eval.py gốc (không sửa) bằng subprocess và lưu đầu ra.

    Input : args (list[str] tham số cho eval.py), out_md (file .md lưu stdout). Output: stdout (str).
    """
    cmd = [sys.executable, str(train.REPO_ROOT / "eval.py"), *args]
    # PYTHONUTF8=1: eval.py ghi grade_I.json (có tiếng Việt) bằng write_text không chỉ định encoding; trên Windows
    # mặc định là cp1252 nên lỗi. Chế độ UTF-8 của Python sửa việc này mà không phải sửa eval.py.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    text = r.stdout + (("\n[stderr]\n" + r.stderr) if r.returncode else "")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(f"```\n$ python eval.py {' '.join(args)}\n```\n\n{text}", encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"eval.py lỗi:\n{text}")
    return r.stdout


def confusion_figure(pattern: str, labels_csv: str, path: Path) -> None:
    """[Implemented by Claude (AI assistant)] Ma trận nhầm lẫn trên test (cộng qua các seed), số lượng + % theo hàng.

    Input : pattern (glob file dự đoán), labels_csv, path (.png). Output: None.
    """
    import glob

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(train.REPO_ROOT))
    import eval as ev

    cm = sum(ev.confusion_matrix(p.y_true, p.y_pred) for p in (ev.read_pred(f) for f in sorted(glob.glob(pattern))))
    rows = cm / cm.sum(1, keepdims=True)
    names = ev.load_names(labels_csv)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}\n{rows[i, j]:.1%}", ha="center", va="center", fontsize=7,
                        color="white" if rows[i, j] > 0.5 else "black")
    ax.set_xticks(range(9), names, rotation=35, ha="right"); ax.set_yticks(range(9), names)
    ax.set_xlabel("nhãn dự đoán"); ax.set_ylabel("nhãn thật")
    ax.set_title("F01 trên test - ma trận nhầm lẫn (cộng các seed; % theo hàng = recall)")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


def error_examples(pred_csv: Path, images_dir: str, path: Path, pairs=((0, 7), (7, 0)), n: int = 6) -> dict:
    """[Implemented by Claude (AI assistant)] Ảnh bị đoán sai cho các cặp (thật -> dự đoán), mặc định Chinee <-> Snake.

    Input : pred_csv (file dự đoán test của một seed), images_dir, path (.png), pairs, n (ảnh mỗi cặp).
    Output: dict {"0->7": [tên file...], "7->0": [...]} để dẫn trong báo cáo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    df = pd.read_csv(pred_csv)
    fig, axes = plt.subplots(len(pairs), n, figsize=(2.2 * n, 2.5 * len(pairs)))
    out = {}
    for r, (t, p) in enumerate(pairs):
        sub = df[(df["y_true"] == t) & (df["y_pred"] == p)].sort_values(f"p{p}", ascending=False).head(n)
        out[f"{t}->{p}"] = sub["Filename"].tolist()
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(sub):
                row = sub.iloc[c]
                ax.imshow(Image.open(Path(images_dir) / row["Filename"]).convert("RGB"))
                ax.set_title(f"thật {ds_mod.CLASS_NAMES[t][:11]}\nđoán {ds_mod.CLASS_NAMES[p][:11]} ({row[f'p{p}']:.2f})",
                             fontsize=7)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=100); plt.close(fig)
    return out


def main() -> None:
    """[Implemented by Claude (AI assistant)] Toàn bộ Bước 4.

    Input : dòng lệnh --force (chỉ khi thật sự phải chạy lại test; ghi vào báo cáo).
    Output: None. Luồng: kiểm tra TEST_DONE -> với mỗi seed: T00 (1 view, T=1) và F01 (phương pháp đã chọn,
      T khớp trên VAL của chính seed đó) -> lưu predictions -> đo độ trễ batch 1 của pipeline F01 ->
      eval.py score (F01, T00, F01_uncal) + grade -> hình ma trận nhầm lẫn + ảnh lỗi -> ghi TEST_DONE.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if DONE.exists() and not args.force:
        raise SystemExit(f"Test đã chạy ({DONE}). Quy tắc: test một lần mỗi seed. Dùng --force nếu bắt buộc "
                         "và ghi rõ lý do trong báo cáo.")
    dec = ex.load_decisions()
    infd = dec["inference"]
    method, space, size = infd["method"], infd["space"], infd["img_size"]
    pred_dir = Path("predictions")
    temps = {}
    for seed in ex.FINAL_SEEDS:
        # --- mốc T00 + I00 ---
        c0 = ex.training_cfg("T00", seed)
        dev = train.resolve_device(c0.device)
        for split in ("val", "test"):
            names, y, z = predict_split(c0, split, "none", "prob", c0.img_size, dev)
            save_predictions(pred_dir / f"T00_seed{seed}_{split}.csv", names, y, train.softmax_np(z))
        # --- chung kết F01: suy luận đã chọn + temperature scaling (T khớp trên VAL) ---
        cf = ex.final_cfg(seed)
        nv, yv, zv = predict_split(cf, "val", method, space, size, dev)
        T = inf.fit_temperature(zv, yv)
        temps[f"seed{seed}"] = T
        save_predictions(pred_dir / f"F01_seed{seed}_val.csv", nv, yv, inf.apply_temperature(zv, T))
        nt, yt, zt = predict_split(cf, "test", method, space, size, dev)
        save_predictions(pred_dir / f"F01_uncal_seed{seed}_test.csv", nt, yt, inf.apply_temperature(zt, 1.0))
        save_predictions(pred_dir / f"F01_seed{seed}_test.csv", nt, yt, inf.apply_temperature(zt, T))
        print(f"[final] seed {seed}: T = {T:.3f}", flush=True)

    # --- độ trễ batch 1 của pipeline F01 (đúng cách: warmup + synchronize, 100 lần) ---
    cf = ex.final_cfg(0)
    dev = train.resolve_device(cf.device)
    model = train.load_trained_model(cf, dev)
    k = views_for(method, size)[2]
    r = benchmark.latency_report(model, 1, size, device=dev.type, iters=100) if k == 1 else \
        benchmark.tta_latency(model, k, batch_size=1, img_size=size, device=dev.type, iters=100)
    latency = {"pipeline": f"F01 {cf.backbone} {method} K={k} {size}px", "device": r["gpu"], "p50": r["p50"],
               "p95": r["p95"], "p99": r["p99"]}
    Path("results").mkdir(exist_ok=True)
    Path("results/final_temperatures.json").write_text(json.dumps({"temperatures": temps, "latency_batch1": latency,
                                                                    "inference": infd}, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")

    # --- chấm bằng eval.py gốc ---
    lab = Path(ex.BASE.labels_dir)
    common = ["--test-csv", str(lab / "test_subset0.csv"), "--labels", str(lab / "labels.csv")]
    ev_dir = Path("results/eval")
    for tag, pat in (("F01", "predictions/F01_seed*_test.csv"), ("T00", "predictions/T00_seed*_test.csv"),
                     ("F01_uncal", "predictions/F01_uncal_seed*_test.csv")):
        print(run_eval(["score", "--pred", pat, *common, "--tag", tag, "--out", str(ev_dir)], ev_dir / f"score_{tag}.md"))
    print(run_eval(["grade", "--final", "predictions/F01_seed*_test.csv", "--baseline", "predictions/T00_seed*_test.csv",
                    "--uncal", "predictions/F01_uncal_seed*_test.csv", "--final-val", "predictions/F01_seed*_val.csv",
                    "--val-csv", str(lab / "val_subset0.csv"), "--latency-p95-ms", f"{latency['p95']:.2f}",
                    "--latency-method", "proper", *common, "--out", str(ev_dir)], ev_dir / "grade.md"))

    confusion_figure("predictions/F01_seed*_test.csv", str(lab / "labels.csv"), Path("figures/confusion_test_F01.png"))
    errs = error_examples(pred_dir / "F01_seed0_test.csv", ex.BASE.images_dir, Path("figures/errors_chinee_snake.png"))
    DONE.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "forced": args.force,
                                "error_examples": errs}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
