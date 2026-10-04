"""eda.py - Bước 0 của GUIDE: EDA, kiểm tra chia dữ liệu và kiểm tra pipeline trước khi chạy thật.

[Implemented by Claude (AI assistant)]
Chạy từ thư mục bài nộp (submissions/<mssv>_<ten>/):
    python code/eda.py --images ../../images --labels ../../data/labels
Ghi ra:
    figures/eda_class_distribution.png   biểu đồ cột số ảnh mỗi lớp theo train/val/test
    figures/eda_samples.png              3 ảnh mỗi lớp
    figures/eda_augmentations.png        ảnh sau augmentation (đã giải chuẩn hoá) kèm nhãn
    figures/eda_cutmix_mixup.png         ảnh sau CutMix và Mixup kèm lam
    results/eda.json                     số liệu: kiểm tra split, đối chiếu Table 1, thống kê ảnh,
                                         loss ban đầu, overfit một batch
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as ds_mod  # noqa: E402
import losses as loss_mod  # noqa: E402
import model as model_mod  # noqa: E402
import train  # noqa: E402

# Table 1 của bài báo DeepWeeds (README mục 2), theo thứ tự Label 0..8
PAPER_TABLE1 = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]


def class_distribution(split_report: dict, out_png: Path) -> dict:
    """[Implemented by Claude (AI assistant)] Vẽ phân bố lớp và đối chiếu với Table 1 của bài báo.

    Input : split_report (dict do dataset.check_split trả về, dùng khoá "per_class"); out_png (Path).
    Output: dict {"table": list[dict] mỗi lớp {"class", "train", "val", "test", "all", "paper", "diff"},
                  "imbalance_ratio": float (lớp lớn nhất / lớp nhỏ nhất), "all_match_paper": bool}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = split_report["per_class"]
    rows = []
    for i, c in enumerate(ds_mod.CLASS_NAMES):
        rows.append({"class": c, "train": pc["train"][c], "val": pc["val"][c], "test": pc["test"][c],
                     "all": pc["all"][c], "paper": PAPER_TABLE1[i], "diff": pc["all"][c] - PAPER_TABLE1[i]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(df))
    for j, (split, color) in enumerate((("train", "#4C72B0"), ("val", "#DD8452"), ("test", "#55A868"))):
        bars = ax.bar(x + (j - 1) * 0.27, df[split], width=0.27, label=split, color=color)
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x, df["class"], rotation=20)
    ax.set_yscale("log")
    ax.set_ylabel("số ảnh (thang log)")
    ax.set_title("DeepWeeds fold 0: số ảnh mỗi lớp theo tập (Negatives ≈ 52%)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"table": rows, "imbalance_ratio": float(df["all"].max() / df["all"].min()),
            "all_match_paper": bool((df["diff"] == 0).all())}


def sample_grid(df: pd.DataFrame, images_dir: Path, out_png: Path, per_class: int = 3, seed: int = 0) -> None:
    """[Implemented by Claude (AI assistant)] Lưới per_class ảnh ngẫu nhiên cho mỗi lớp (đọc thẳng file JPEG).

    Input : df (DataFrame Filename, Label - dùng tập TRAIN), images_dir, out_png, per_class (int), seed (int).
    Output: None; ghi ảnh lưới 9 hàng x per_class cột.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(9, per_class, figsize=(2.2 * per_class, 2.1 * 9))
    for c in range(9):
        names = df[df["Label"] == c]["Filename"].to_numpy()
        for j, name in enumerate(rng.choice(names, per_class, replace=False)):
            ax = axes[c, j]
            ax.imshow(Image.open(images_dir / name).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(ds_mod.CLASS_NAMES[c], fontsize=9)
    fig.suptitle("3 ảnh ngẫu nhiên mỗi lớp (tập train)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def image_stats(df: pd.DataFrame, images_dir: Path, n: int = 500, seed: int = 0) -> dict:
    """[Implemented by Claude (AI assistant)] Thống kê kích thước/kênh và mean/std pixel trên n ảnh train.

    Input : df (train), images_dir, n (số ảnh lấy mẫu), seed.
    Output: dict {"sizes": {"256x256": count, ...}, "modes": {"RGB": count}, "mean": [3], "std": [3], "n": n}
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    sizes, modes, sums, sqs, cnt = {}, {}, np.zeros(3), np.zeros(3), 0
    for name in rng.choice(df["Filename"].to_numpy(), n, replace=False):
        with Image.open(images_dir / name) as im:
            sizes[f"{im.size[0]}x{im.size[1]}"] = sizes.get(f"{im.size[0]}x{im.size[1]}", 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        sums += a.reshape(-1, 3).sum(0)
        sqs += (a.reshape(-1, 3) ** 2).sum(0)
        cnt += a.shape[0] * a.shape[1]
    mean = sums / cnt
    std = np.sqrt(sqs / cnt - mean ** 2)
    return {"sizes": sizes, "modes": modes, "mean": mean.round(4).tolist(), "std": std.round(4).tolist(), "n": n}


def _denorm(x: torch.Tensor) -> np.ndarray:
    """[Implemented by Claude (AI assistant)] Giải chuẩn hoá tensor (3,H,W) về ảnh HxWx3 trong [0,1] để vẽ."""
    m = torch.tensor(ds_mod.IMAGENET_MEAN)[:, None, None]
    s = torch.tensor(ds_mod.IMAGENET_STD)[:, None, None]
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def augmentation_grid(cfg: train.Config, train_df: pd.DataFrame, out_png: Path, out_mix_png: Path) -> None:
    """[Implemented by Claude (AI assistant)] Vẽ ảnh SAU augmentation (đã giải chuẩn hoá) cùng nhãn, và CutMix/Mixup.

    Input : cfg (img_size, đường dẫn), train_df, out_png (lưới các mức aug), out_mix_png (CutMix + Mixup).
    Output: None. Mục đích: kiểm tra ảnh và nhãn khớp nhau sau toàn bộ pipeline (GUIDE 1.3 ý 4).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    np.random.seed(0)
    augs = ["eval", "basic", "flipv", "color", "trivial", "randaug"]
    pick = train_df.groupby("Label").head(1).sort_values("Label").head(6)
    fig, axes = plt.subplots(len(pick), len(augs), figsize=(2.1 * len(augs), 2.2 * len(pick)))
    for r, (_, row) in enumerate(pick.iterrows()):
        for c, aug in enumerate(augs):
            tf = ds_mod.build_transforms(aug != "eval", cfg.img_size, "basic" if aug == "eval" else aug)
            ds = ds_mod.DeepWeedsDataset(pd.DataFrame([row]), cfg.images_dir, tf)
            x, y, _ = ds[0]
            ax = axes[r, c]
            ax.imshow(_denorm(x)); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(aug, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{y}: {ds_mod.CLASS_NAMES[y]}", fontsize=8)
    fig.suptitle(f"Ảnh sau transform (img_size={cfg.img_size}), nhãn ở cột trái")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)

    tf = ds_mod.build_transforms(True, cfg.img_size, "basic")
    sub = train_df.groupby("Label").head(1).sort_values("Label")
    ds = ds_mod.DeepWeedsDataset(sub, cfg.images_dir, tf)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
    for r, mode in enumerate(("cutmix", "mixup")):
        xm, (ya, yb, lam) = loss_mod.mix_batch(x, y, alpha=1.0, mode=mode)
        for c in range(6):
            ax = axes[r, c]
            ax.imshow(_denorm(xm[c])); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{mode} lam={lam:.2f}\n{ds_mod.CLASS_NAMES[int(ya[c])][:12]} + "
                         f"{ds_mod.CLASS_NAMES[int(yb[c])][:12]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_mix_png, dpi=90)
    plt.close(fig)


def pipeline_checks(cfg: train.Config, train_df: pd.DataFrame, overfit_steps: int = 80) -> dict:
    """[Implemented by Claude (AI assistant)] Kiểm tra pipeline theo checklist slide trang 59 (GUIDE 1.3).

    Input : cfg (backbone, img_size, lr...), train_df, overfit_steps (int).
    Output: dict {"backbone", "initial_loss": float (head khởi tạo 0, cách dùng khi train),
                  "initial_loss_timm_default_head": float (head mặc định của timm, để so sánh),
                  "expected_initial_loss": ln 9 = 2.1972,
                  "overfit_batch_size": 16, "overfit_loss_start": float, "overfit_loss_end": float,
                  "overfit_acc_end": float, "overfit_curve": list[float] (loss mỗi 10 bước),
                  "train_eval_mode_ok": bool}
    Cách làm: (1) loss CE của model tiền huấn luyện + head mới trên 2 batch train ở chế độ eval phải ~ ln 9;
      (2) 16 ảnh cố định (transform eval, không augmentation), AdamW lr 1e-3 cho mọi tham số, overfit tới loss ~0;
      (3) kiểm tra model.eval() tắt train mode của mọi BN và set_train_mode bật lại.
    """
    train.set_seed(0)
    dev = train.resolve_device(cfg.device)  # GPU nếu có
    m = model_mod.build_model(cfg.backbone, pretrained=True, num_classes=9, img_size=cfg.img_size).to(dev)
    ce = nn.CrossEntropyLoss()
    tf_eval = ds_mod.build_transforms(False, cfg.img_size)
    loader = ds_mod.make_loader(train_df, cfg.images_dir, tf_eval, 64, train=True, num_workers=0,
                                preload_size=ds_mod.eval_resize(cfg.img_size), cache_dir=cfg.cache_dir, seed=0)
    it = iter(loader)
    m.eval()
    batches = [next(it), next(it)]
    with torch.no_grad():
        init_losses = [float(ce(m(x.to(dev)), y.to(dev))) for x, y, _ in batches]
        m_def = model_mod.build_model(cfg.backbone, pretrained=True, num_classes=9, img_size=cfg.img_size,
                                      head_init="default").to(dev).eval()
        init_losses_default = [float(ce(m_def(x.to(dev)), y.to(dev))) for x, y, _ in batches]
        del m_def

    xb, yb, _ = next(it)
    xb, yb = xb[:16].to(dev), yb[:16].to(dev)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    curve = []
    for step in range(overfit_steps):
        loss = ce(m(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0 or step == overfit_steps - 1:
            curve.append(round(float(loss), 4))
    m.eval()
    with torch.no_grad():
        logits = m(xb)
        end_loss, end_acc = float(ce(logits, yb)), float((logits.argmax(1) == yb).float().mean())
    bns = [mod for mod in m.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
    eval_ok = all(not b.training for b in bns)
    train.set_train_mode(m)
    train_ok = all(b.training for b in bns)
    return {"backbone": cfg.backbone, "img_size": cfg.img_size,
            "initial_loss": float(np.mean(init_losses)), "expected_initial_loss": math.log(9),
            "initial_loss_timm_default_head": float(np.mean(init_losses_default)),
            "overfit_batch_size": 16, "overfit_steps": overfit_steps,
            "overfit_loss_start": curve[0], "overfit_loss_end": end_loss, "overfit_acc_end": end_acc,
            "overfit_curve": curve, "train_eval_mode_ok": bool(eval_ok and train_ok)}


def run_eda(images: str, labels: str, img_size: int, backbones: list[str], cache_dir: str = "runs/cache") -> dict:
    """[Implemented by Claude (AI assistant)] Chạy toàn bộ EDA + kiểm tra pipeline, ghi figures/ và results/eda.json.

    Input : images (thư mục ảnh), labels (thư mục CSV), img_size (int), backbones (list tên timm để kiểm tra
            pipeline), cache_dir (bộ đệm ảnh). Output: dict {"split_check", "class_distribution", "image_stats",
            "pipeline_checks": list[dict]} (cũng ghi ra results/eda.json). Bỏ qua nếu results/eda.json đã có.
    """
    out_json = Path("results/eda.json")
    if out_json.exists():
        print(f"[eda] đã có {out_json}, bỏ qua")
        return json.loads(out_json.read_text(encoding="utf-8"))
    figs = Path("figures")
    cfg = train.Config(images_dir=images, labels_dir=labels, img_size=img_size, cache_dir=cache_dir)
    train_df, val_df, test_df = ds_mod.load_split(labels, 0)
    split = ds_mod.check_split(train_df, val_df, test_df, images)
    res = {"split_check": split}
    res["class_distribution"] = class_distribution(split, figs / "eda_class_distribution.png")
    sample_grid(train_df, Path(images), figs / "eda_samples.png")
    res["image_stats"] = image_stats(train_df, Path(images))
    augmentation_grid(cfg, train_df, figs / "eda_augmentations.png", figs / "eda_cutmix_mixup.png")
    res["pipeline_checks"] = []
    for b in backbones:
        c = train.Config(images_dir=images, labels_dir=labels, img_size=img_size, backbone=b, cache_dir=cache_dir)
        r = pipeline_checks(c, train_df)
        print(f"[pipeline] {b}: initial loss {r['initial_loss']:.3f} (head timm mặc định "
              f"{r['initial_loss_timm_default_head']:.3f}; ln9={math.log(9):.3f}), "
              f"overfit 16 ảnh: {r['overfit_loss_start']:.3f} -> {r['overfit_loss_end']:.4f}, acc {r['overfit_acc_end']:.2f}")
        res["pipeline_checks"].append(r)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def main() -> None:
    """[Implemented by Claude (AI assistant)] Điểm vào dòng lệnh, gọi run_eda.

    Input : --images, --labels, --img-size, --backbones. Output: None (in tóm tắt).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="../../images")
    ap.add_argument("--labels", default="../../data/labels")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--backbones", nargs="*", default=["resnet50", "mobilenetv3_large_100"])
    args = ap.parse_args()
    res = run_eda(args.images, args.labels, args.img_size, args.backbones)
    print(json.dumps({k: v for k, v in res.items() if k != "split_check"}, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
