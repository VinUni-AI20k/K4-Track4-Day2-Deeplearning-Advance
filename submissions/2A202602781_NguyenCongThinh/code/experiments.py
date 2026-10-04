"""experiments.py - danh sách MỌI thí nghiệm huấn luyện (B, T, F) và trình chạy hàng loạt.

[Implemented by Claude (AI assistant)]
Mọi thí nghiệm đi qua MỘT hàm train.run(Config(...)) (RUBRIC mục H); file này chỉ khai báo các Config.
Các lựa chọn giữa các bước (backbone đi tiếp, kết hợp công thức, cấu hình chung kết, phương pháp suy luận)
KHÔNG viết cứng ở đây mà đọc từ results/decisions.json do run_all.py ghi ra (luật chọn chỉ dùng VAL),
và có thể ghi đè bằng tay trong results/decisions_override.json.

Chạy từ thư mục bài nộp (submissions/<mssv>_<ten>/):
    python code/experiments.py --list                 # in danh sách khoá
    python code/experiments.py B01 B02                # chạy vài khoá (bỏ qua cái đã xong, tự resume)
Thường không gọi trực tiếp mà qua run_all.py.

Hai "profile" (tự chọn theo phần cứng, hoặc đặt biến môi trường LAB_PROFILE=gpu|cpu):
    gpu : công thức nền đúng GUIDE 1.4 (224 px, 12 epoch, batch 64, AMP), 6 backbone gợi ý của GUIDE 2.1
    cpu : bản rút gọn cho máy không có GPU (128 px, 8 epoch, AMP tắt), 5 backbone nhỏ
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train  # noqa: E402
from train import Config  # noqa: E402

# ---------------------------------------------------------------------------------------------------------
# Profile phần cứng
# ---------------------------------------------------------------------------------------------------------
PROFILES = {
    "gpu": dict(
        img_size=224, epochs=12, batch_size=64, amp=True, num_workers=4, channels_last=True,
        backbones={
            "B01": ("resnet50", "resnet50"),                        # ResNet (mốc)
            "B02": ("convnext_tiny", "convnext_tiny"),              # ConvNeXt
            "B03": ("deit_small_patch16_224", "deit_small"),        # transformer (DeiT)
            "B04": ("swin_tiny_patch4_window7_224", "swin_tiny"),   # transformer (Swin)
            "B05": ("efficientnet_b0", "efficientnet_b0"),          # mạng nhẹ
            "B06": ("mobilenetv3_large_100", "mobilenetv3"),        # mạng nhẹ
        },
    ),
    "cpu": dict(
        img_size=128, epochs=8, batch_size=64, amp=False, num_workers=0, channels_last=True,
        backbones={
            "B01": ("resnet18", "resnet18"),
            "B02": ("convnext_atto", "convnext_atto"),
            "B03": ("deit_tiny_patch16_224", "deit_tiny"),
            "B04": ("mobilenetv3_large_100", "mobilenetv3"),
            "B05": ("efficientnet_b0", "efficientnet_b0"),
        },
    ),
    # chỉ để kiểm tra code chạy thông suốt từ đầu tới cuối (số liệu vô nghĩa, KHÔNG dùng cho báo cáo)
    "smoke": dict(
        img_size=64, epochs=2, batch_size=32, amp=False, num_workers=0, channels_last=False, limit_train_batches=5,
        backbones={"B01": ("resnet18", "resnet18"), "B02": ("convnext_atto", "convnext_atto"),
                   "B03": ("deit_tiny_patch16_224", "deit_tiny"), "B04": ("mobilenetv3_large_100", "mobilenetv3")},
    ),
}
PROFILE = os.environ.get("LAB_PROFILE") or ("gpu" if torch.cuda.is_available() else "cpu")
P = PROFILES[PROFILE]
BACKBONES = P["backbones"]

# Công thức nền T00 (GUIDE 1.4): AdamW, LR backbone 1e-4 / head 1e-3, wd 0.05 (không cho norm/bias),
# warmup 1 epoch + cosine, CE, chọn checkpoint theo macro-F1 val. Device "auto" = dùng GPU (cuda) nếu có.
BASE = Config(
    exp_id="T00", seed=0, img_size=P["img_size"], epochs=P["epochs"],
    batch_size=int(os.environ.get("LAB_BATCH", P["batch_size"])),  # LAB_BATCH: giảm nếu hết VRAM (ghi vào báo cáo)
    amp=P["amp"], num_workers=P["num_workers"], channels_last=P["channels_last"], device="auto",
    images_dir=os.environ.get("LAB_IMAGES", "../../images"),
    labels_dir=os.environ.get("LAB_LABELS", "../../data/labels"),
    out_dir="runs", pred_dir="predictions", curves_dir="curves", cache_dir="runs/cache",
    limit_train_batches=P.get("limit_train_batches"),
)

# Bước 2: mỗi thí nghiệm khác T00 đúng MỘT yếu tố (trục A-F của GUIDE 3): (trục, mô tả, ghi đè Config)
TRAINING = {
    "T01": ("A", "init=scratch (khởi tạo ngẫu nhiên)", dict(init="scratch")),
    "T02": ("A", "init=frozen (đóng băng backbone, chỉ train head)", dict(init="frozen")),
    "T03": ("B", "aug=trivial (TrivialAugmentWide)", dict(aug="trivial")),
    "T04": ("B", "mix=cutmix (alpha=1)", dict(mix="cutmix", mix_alpha=1.0)),
    "T05": ("B", "aug=flipv (thêm lật dọc)", dict(aug="flipv")),
    "T06": ("C", "loss=ls (label smoothing eps=0.1)", dict(loss="ls", label_smoothing=0.1)),
    "T07": ("C", "loss=focal (gamma=2)", dict(loss="focal", focal_gamma=2.0)),
    "T08": ("C", "loss=ce_weighted (trọng số 1/n_c)", dict(loss="ce_weighted")),
    "T09": ("D", "sampler=balanced (oversample lớp hiếm)", dict(sampler="balanced")),
    "T10": ("E", "lr_head = lr_backbone = 1e-4 (không x10)", dict(lr_head=1e-4)),
    "T11": ("F", "ema_decay=0.999 (đánh giá bằng trọng số EMA)", dict(ema_decay=0.999)),
}
FINAL_SEEDS = (0, 1, 2)
DECISIONS_FILE = Path("results/decisions.json")
OVERRIDE_FILE = Path("results/decisions_override.json")


def load_decisions() -> dict:
    """[Implemented by Claude (AI assistant)] Đọc các lựa chọn giữa các bước.

    Input : không (đọc results/decisions.json rồi ghi đè bằng results/decisions_override.json nếu có).
    Output: dict, các khoá có thể có:
      {"profile": "gpu", "chosen_backbone": "B01",            # Bước 1 -> 2
       "combo": {"aug": "trivial", "loss": "ls", ...},         # T12 = T00 + các ghi đè này
       "final_recipe": {...},                                   # ghi đè Config cho F01
       "final_source": "T12",                                   # run seed 0 tương ứng final_recipe
       "inference": {"method": "hflip", "space": "prob", "img_size": 224},   # Bước 3 -> 4
       "reasons": {...}}                                        # lý do (dán vào báo cáo)
    """
    d = json.loads(DECISIONS_FILE.read_text(encoding="utf-8")) if DECISIONS_FILE.exists() else {}
    if OVERRIDE_FILE.exists():
        d.update(json.loads(OVERRIDE_FILE.read_text(encoding="utf-8")))
    return d


def save_decision(**kw) -> dict:
    """[Implemented by Claude (AI assistant)] Ghi thêm/đè khoá vào results/decisions.json. Output: dict sau khi ghi."""
    d = json.loads(DECISIONS_FILE.read_text(encoding="utf-8")) if DECISIONS_FILE.exists() else {}
    reasons = {**d.get("reasons", {}), **kw.pop("reasons", {})}
    d.update(kw, reasons=reasons, profile=PROFILE)
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DECISIONS_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def backbone_cfg(exp_id: str) -> Config:
    """[Implemented by Claude (AI assistant)] Config của một thí nghiệm backbone B0x.

    Input : exp_id (khoá của BACKBONES). Output: Config = BASE với backbone/desc tương ứng, seed 0.
    """
    name, desc = BACKBONES[exp_id]
    return replace(BASE, exp_id=exp_id, backbone=name, desc=desc)


def chosen_backbone() -> tuple[str, str]:
    """[Implemented by Claude (AI assistant)] (tên timm, tên ngắn) của backbone đã chọn ở Bước 1.

    Output: tuple[str, str]. Lỗi rõ ràng nếu chưa chạy bước chọn (run_all.py choose_backbone).
    """
    key = load_decisions().get("chosen_backbone")
    if key is None:
        raise RuntimeError("chưa chọn backbone: chạy `python code/run_all.py choose_backbone` sau Bước 1")
    return BACKBONES[key]


def training_cfg(exp_id: str, seed: int = 0) -> Config:
    """[Implemented by Claude (AI assistant)] Config của T00..T12 (T00 = công thức nền trên backbone đã chọn).

    Input : exp_id ("T00", khoá của TRAINING, hoặc "T12" = kết hợp), seed (int).
    Output: Config; desc là mô tả không dấu cách cho tên ảnh curves/<exp_id>_<desc>.png.
    """
    name, short = chosen_backbone()
    base = replace(BASE, backbone=name, seed=seed)
    if exp_id == "T00":
        return replace(base, exp_id="T00", desc=f"baseline_{short}")
    if exp_id == "T12":
        overrides = load_decisions().get("combo") or {}
        if not overrides:
            raise RuntimeError("chưa có combo: chạy `python code/run_all.py choose_combo`")
    else:
        overrides = TRAINING[exp_id][2]
    return replace(base, exp_id=exp_id, desc=_desc(overrides) if exp_id != "T12" else "combo_" + _desc(overrides),
                   **overrides)


def _desc(overrides: dict) -> str:
    """[Implemented by Claude (AI assistant)] Mô tả ngắn từ dict ghi đè, ví dụ {"loss": "ls"} -> "loss-ls"."""
    skip = ("label_smoothing", "focal_gamma", "mix_alpha")
    return "_".join(f"{k}-{v}" for k, v in overrides.items() if k not in skip).replace(".", "p")


def final_cfg(seed: int) -> Config:
    """[Implemented by Claude (AI assistant)] Config chung kết F01 (backbone đã chọn + final_recipe) cho một seed.

    Input : seed (int). Output: Config exp_id="F01".
    """
    name, short = chosen_backbone()
    recipe = load_decisions().get("final_recipe")
    if recipe is None:
        raise RuntimeError("chưa chốt công thức chung kết: chạy `python code/run_all.py choose_final`")
    return replace(BASE, exp_id="F01", backbone=name, seed=seed, desc=f"final_{short}", **recipe)


def get_cfg(key: str) -> Config:
    """[Implemented by Claude (AI assistant)] Config theo khoá: "B01", "T00", "T07", "T12", "F01_seed1", "T00_seed2".

    Input : key (str). Output: Config. Được run_all.py, infer_experiments.py, final.py, make_results.py dùng.
    """
    if key in BACKBONES:
        return backbone_cfg(key)
    if "_seed" in key:
        exp, seed = key.split("_seed")
        return final_cfg(int(seed)) if exp == "F01" else training_cfg(exp, int(seed))
    return training_cfg(key)


def stage_keys(stage: str) -> list[str]:
    """[Implemented by Claude (AI assistant)] Các khoá của một nhóm.

    Input : "backbones" | "training" | "combo" | "final". Output: list[str].
    """
    if stage == "backbones":
        return list(BACKBONES)
    if stage == "training":
        return ["T00", *TRAINING]
    if stage == "combo":
        return ["T12"]
    if stage == "final":
        return [f"F01_seed{s}" for s in FINAL_SEEDS] + [f"T00_seed{s}" for s in FINAL_SEEDS if s != 0]
    raise ValueError(stage)


def reuse_identical_run(cfg: Config) -> bool:
    """[Implemented by Claude (AI assistant)] Dùng lại một lần chạy đã xong có cấu hình Y HỆT (chỉ khác exp_id/desc).

    Input : cfg (Config định chạy). Output: bool - True nếu đã chép kết quả, không cần train lại.
    Cách làm: T00 seed 0 trùng hoàn toàn với B0x của backbone đã chọn (cùng công thức, cùng seed), và F01 seed 0
      trùng run seed 0 của công thức đã chọn (ví dụ T12). Chép thư mục run, đổi exp_id/config trong summary,
      vẽ lại biểu đồ với tên mới (ghi chú "= <exp gốc>"). Báo cáo nêu rõ việc dùng lại này.
    """
    import pandas as pd

    key = train._cfg_key({**dataclasses.asdict(cfg), "exp_id": "", "desc": ""})
    candidates = [*BACKBONES, "T00", *TRAINING, "T12"]
    for k in candidates:
        try:
            other = get_cfg(k)
        except RuntimeError:
            continue
        if other.exp_id == cfg.exp_id:
            continue
        src = train.run_dir(other)
        if not (src / "summary.json").exists():
            continue
        if train._cfg_key({**dataclasses.asdict(other), "exp_id": "", "desc": ""}) != key:
            continue
        dst = train.run_dir(cfg)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        s = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
        s.update(config=dataclasses.asdict(cfg), exp_id=cfg.exp_id, reused_from=other.exp_id)
        cpath = train.curve_path(cfg)
        train.plot_curves(pd.read_csv(dst / "history.csv").to_dict("records"), cpath,
                          f"{cfg.exp_id} seed{cfg.seed} - {cfg.backbone} ({cfg.desc}) [cùng run với {other.exp_id}]")
        s["curve"] = str(cpath)
        (dst / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[experiments] {cfg.exp_id} seed{cfg.seed} trùng cấu hình {other.exp_id}: dùng lại kết quả")
        return True
    return False


def run_keys(keys: list[str]) -> list[dict]:
    """[Implemented by Claude (AI assistant)] Chạy tuần tự các khoá; bỏ qua run đã xong; dùng lại run trùng cấu hình.

    Input : keys (list[str]). Output: list[dict] summary của từng run (đọc từ summary.json).
    """
    out = []
    for k in keys:
        cfg = get_cfg(k)
        t0 = time.time()
        done = (train.run_dir(cfg) / "summary.json").exists()
        if not done and reuse_identical_run(cfg):
            res = json.loads((train.run_dir(cfg) / "summary.json").read_text(encoding="utf-8"))
        else:
            res = train.run(cfg)
        print(f"[experiments] {k}: val macro-F1 {res['val_macro_f1']:.4f}, top-1 {res['val_top1']:.4f}, "
              f"best ep {res['best_epoch']}, {(time.time() - t0) / 60:.1f} phút", flush=True)
        out.append(res)
    return out


def main() -> None:
    """[Implemented by Claude (AI assistant)] CLI: --list hoặc danh sách khoá cần chạy. Output: None."""
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.keys:
        print(f"profile = {PROFILE}: {P['img_size']} px, {P['epochs']} epoch, amp={P['amp']}")
        for stage in ("backbones", "training", "combo", "final"):
            print(stage, stage_keys(stage))
        return
    run_keys(args.keys)


if __name__ == "__main__":
    main()
