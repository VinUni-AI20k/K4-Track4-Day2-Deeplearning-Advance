"""run_all.py - điều phối toàn bộ bài lab theo đúng thứ tự của GUIDE, từng giai đoạn chạy lại được (resume).

[Implemented by Claude (AI assistant)]
Chạy từ thư mục bài nộp (submissions/<mssv>_<ten>/):
    python code/run_all.py all                    # chạy hết từ đầu tới cuối (bỏ qua phần đã xong)
    python code/run_all.py backbones              # hoặc từng giai đoạn
Các giai đoạn (theo thứ tự):
    eda             Bước 0: EDA, kiểm tra split, loss ban đầu ~ ln 9, overfit 1 batch, ảnh sau augmentation
    backbones       Bước 1: train B01..B0n (công thức nền, seed 0)
    choose_backbone Bước 1: đo độ trễ sơ bộ, CHỌN backbone đi tiếp (luật dưới, chỉ dùng val)
    training        Bước 2: T00 + T01..T11 (mỗi run khác T00 một yếu tố)
    choose_combo    Bước 2: chọn các yếu tố thắng để kết hợp -> T12
    combo           Bước 2: train T12
    choose_final    Bước 2 -> 4: chốt công thức chung kết (T00 / T tốt nhất / T12, theo macro-F1 val)
    inference       Bước 3: các phương pháp suy luận trên val + độ trễ, chọn phương pháp cho F01
    final_train     Bước 4: train F01 seed 0/1/2 và T00 seed 1/2
    final_test      Bước 4: test MỘT lần mỗi seed, eval.py score + grade
    results         Bước 5: results.xlsx, bảng markdown, biểu đồ tổng hợp
Mọi lựa chọn được ghi vào results/decisions.json kèm lý do. Muốn chọn khác: viết khoá tương ứng vào
results/decisions_override.json (ví dụ {"chosen_backbone": "B02"}) rồi chạy lại các giai đoạn sau.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import experiments as ex  # noqa: E402
import train  # noqa: E402

CLEAR_DELTA = 0.003      # Δ macro-F1 val tối thiểu để coi một yếu tố là "thắng rõ" (1 seed, xem report)
BACKBONE_TIE = 0.01      # các backbone cách backbone tốt nhất < 0.01 macro-F1 coi như ngang nhau -> chọn cái nhanh hơn


def summary_of(key: str) -> dict:
    """[Implemented by Claude (AI assistant)] Đọc summary.json của run theo khoá. Output: dict (xem train.run)."""
    return json.loads((train.run_dir(ex.get_cfg(key)) / "summary.json").read_text(encoding="utf-8"))


def stage_eda() -> None:
    """[Implemented by Claude (AI assistant)] Bước 0: gọi eda.run_eda với độ phân giải và 2 backbone của profile."""
    import eda
    bbs = [v[0] for v in ex.BACKBONES.values()]
    eda.run_eda(ex.BASE.images_dir, ex.BASE.labels_dir, ex.BASE.img_size, [bbs[0], bbs[-1]], cache_dir=ex.BASE.cache_dir)


def stage_choose_backbone() -> None:
    """[Implemented by Claude (AI assistant)] Đo độ trễ sơ bộ từng backbone và chọn backbone đi tiếp.

    Input : summary của B01..B0n. Output: None; ghi results/backbone_latency.csv và decisions["chosen_backbone"].
    Luật (chỉ val): lấy các backbone có macro-F1 val >= tốt nhất - 0.01 (chênh nhỏ, 1 seed), trong đó chọn cái có
      độ trễ p50 batch 1 thấp nhất -> cân bằng chất lượng và tốc độ (slide trang 33: "đủ tốt trong độ trễ đo được").
    """
    rows = []
    for k in ex.BACKBONES:
        cfg = ex.backbone_cfg(k)
        s = summary_of(k)
        dev = train.resolve_device(cfg.device)
        r = benchmark.latency_report(train.load_trained_model(cfg, dev), 1, cfg.img_size, device=dev.type,
                                     warmup=10, iters=50)
        rows.append({"exp_id": k, "backbone": cfg.backbone, "val_macro_f1": s["val_macro_f1"],
                     "p50_ms": r["p50"], "p95_ms": r["p95"], "device": r["gpu"]})
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/backbone_latency.csv", index=False)
    best = df["val_macro_f1"].max()
    pool = df[df["val_macro_f1"] >= best - BACKBONE_TIE].sort_values("p50_ms")
    pick = pool.iloc[0]
    reason = (f"{pick['exp_id']} ({pick['backbone']}): macro-F1 val {pick['val_macro_f1']:.4f} (tốt nhất {best:.4f}), "
              f"p50 batch 1 = {pick['p50_ms']:.2f} ms; nhóm ngang nhau (cách tốt nhất < {BACKBONE_TIE}): "
              f"{', '.join(pool['exp_id'])} -> chọn cái nhanh nhất")
    ex.save_decision(chosen_backbone=pick["exp_id"], reasons={"backbone": reason})
    print(df.to_string(), "\n", reason)


def training_deltas() -> pd.DataFrame:
    """[Implemented by Claude (AI assistant)] Bảng Δ macro-F1 val của T01..T11 so với T00.

    Output: DataFrame cột exp_id, axis, desc, overrides (dict), val_macro_f1, delta.
    """
    base = summary_of("T00")["val_macro_f1"]
    rows = []
    for k, (axis, desc, ov) in ex.TRAINING.items():
        s = summary_of(k)
        rows.append({"exp_id": k, "axis": axis, "desc": desc, "overrides": ov, "val_macro_f1": s["val_macro_f1"],
                     "delta": s["val_macro_f1"] - base})
    return pd.DataFrame(rows)


def stage_choose_combo() -> None:
    """[Implemented by Claude (AI assistant)] Chọn các yếu tố để kết hợp thành T12 ("tham lam theo trục").

    Luật (chỉ val): mỗi trục lấy giá trị có Δ lớn nhất nếu Δ >= CLEAR_DELTA; kết hợp các trục đó.
      Nếu ít hơn 2 trục đạt, lấy 2 trục có Δ dương lớn nhất (khác trục) để vẫn thử một kết hợp (RUBRIC C);
      nếu vẫn thiếu, lấy 2 trục có Δ lớn nhất. Trục A (scratch/frozen) không đưa vào nếu Δ <= 0.
      Không kết hợp đồng thời sampler cân bằng (T09) và CE có trọng số (T08): giữ cái có Δ lớn hơn.
    Output: None; ghi decisions["combo"] (dict ghi đè Config) và lý do.
    """
    d = training_deltas().sort_values("delta", ascending=False)
    per_axis = d.drop_duplicates("axis")
    conflict = ""
    # sampler cân bằng (T09) và CE có trọng số (T08) cùng bù mất cân bằng lớp: dùng cả hai là bù hai lần
    # -> loại cái có Δ nhỏ hơn khỏi danh sách ứng viên (per_axis đã sắp theo Δ giảm dần)
    if {"T08", "T09"} <= set(per_axis["exp_id"]):
        drop = per_axis[per_axis["exp_id"].isin(["T08", "T09"])].iloc[-1]["exp_id"]
        per_axis = per_axis[per_axis["exp_id"] != drop]
        conflict = f"; loại {drop} khỏi ứng viên vì T08 và T09 cùng bù mất cân bằng (tránh bù hai lần)"
    chosen = per_axis[per_axis["delta"] >= CLEAR_DELTA]
    note = f"các trục có Δ >= {CLEAR_DELTA}"
    if len(chosen) < 2:
        chosen = per_axis[per_axis["delta"] > 0].head(2)
        note = "ít hơn 2 trục thắng rõ: lấy 2 trục có Δ dương lớn nhất"
    if len(chosen) < 2:
        chosen = per_axis[(per_axis["axis"] != "A") | (per_axis["delta"] > 0)].head(2)
        note = "ít hơn 2 trục có Δ dương: lấy 2 trục có Δ lớn nhất (thí nghiệm kiểm tra cộng dồn)"
    note += conflict
    combo = {}
    for ov in chosen["overrides"]:
        combo.update(ov)
    reason = f"{note}: " + ", ".join(f"{r.exp_id} ({r.desc}, Δ={r.delta:+.4f})" for r in chosen.itertuples())
    ex.save_decision(combo=combo, combo_from=list(chosen["exp_id"]), reasons={"combo": reason})
    print(d[["exp_id", "axis", "desc", "val_macro_f1", "delta"]].to_string(), "\n", reason)


def stage_choose_final() -> None:
    """[Implemented by Claude (AI assistant)] Chốt công thức chung kết: T00, T đơn lẻ tốt nhất hoặc T12 (macro-F1 val).

    Output: None; ghi decisions["final_recipe"] (dict ghi đè Config) và decisions["final_source"] (run seed 0).
    """
    d = training_deltas()
    cands = {"T00": ({}, summary_of("T00")["val_macro_f1"])}
    top = d.sort_values("val_macro_f1", ascending=False).iloc[0]
    cands[top["exp_id"]] = (top["overrides"], top["val_macro_f1"])
    cands["T12"] = (ex.load_decisions()["combo"], summary_of("T12")["val_macro_f1"])
    src = max(cands, key=lambda k: cands[k][1])
    reason = "macro-F1 val (seed 0): " + ", ".join(f"{k} {v[1]:.4f}" for k, v in cands.items()) + f" -> {src}"
    ex.save_decision(final_recipe=cands[src][0], final_source=src, reasons={"final": reason})
    print(reason)


def main() -> None:
    """[Implemented by Claude (AI assistant)] CLI chọn giai đoạn. Input: tên giai đoạn hoặc "all". Output: None."""
    stages = {
        "eda": stage_eda,
        "backbones": lambda: ex.run_keys(ex.stage_keys("backbones")),
        "choose_backbone": stage_choose_backbone,
        "training": lambda: ex.run_keys(ex.stage_keys("training")),
        "choose_combo": stage_choose_combo,
        "combo": lambda: ex.run_keys(ex.stage_keys("combo")),
        "choose_final": stage_choose_final,
        "inference": lambda: __import__("infer_experiments").main(),
        "final_train": lambda: ex.run_keys(ex.stage_keys("final")),
        "final_test": lambda: __import__("final").main(),
        "results": lambda: __import__("make_results").main(),
    }
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*stages, "all"])
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    print(f"[run_all] profile = {ex.PROFILE} ({ex.P['img_size']} px, {ex.P['epochs']} epoch, "
          f"device = {train.resolve_device('auto')})", flush=True)
    todo = list(stages) if args.stage == "all" else [args.stage]
    for name in todo:
        if args.stage == "all" and name == "final_test" and Path("results/eval/TEST_DONE.json").exists():
            print("[run_all] final_test đã chạy trước đó, bỏ qua (test chỉ một lần)")
            continue
        print(f"\n========== {name} ==========", flush=True)
        stages[name]()


if __name__ == "__main__":
    main()
