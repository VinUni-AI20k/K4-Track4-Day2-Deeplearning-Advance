# Lab Day 2 – DeepWeeds: bài nộp của Nguyễn Công Thịnh (2A202602781)

Bài nộp gồm code hoàn chỉnh (`code/`, phát triển từ bộ khung `starter/`) và các sản phẩm do code sinh ra:
`results.xlsx`, `report.md`, `curves/`, `predictions/`, `figures/`, `results/`.

> Phần code được viết với sự hỗ trợ của Claude (AI assistant); mọi hàm do AI viết đều có dòng
> `[Implemented by Claude (AI assistant)]` ngay dưới docstring/TODO gốc. Mọi con số trong báo cáo đến từ các lần
> chạy thật trên máy được khai báo ở mục "Môi trường".

## Link chạy lại

- Notebook: [`code/lab_day2.ipynb`](code/lab_day2.ipynb) (chạy được trên Kaggle/Colab hoặc Jupyter ở máy có GPU).
- Link Kaggle/Colab: _(điền sau khi tải notebook lên)_

## Môi trường

| Thành phần | Phiên bản / giá trị |
|---|---|
| Python | 3.11 |
| PyTorch | 2.x bản CUDA (ghi chính xác: xem `runs/<exp>/seed0/config.json`, khoá `env`) |
| timm | 1.0.x |
| GPU | _(điền tên GPU, ví dụ RTX 3060 8 GB; cũng có trong config.json)_ |
| Seed | 0 cho Bước 1–3; 0, 1, 2 cho chung kết `F01` và mốc `T00` |

Cài đặt (Windows/Linux, máy có GPU NVIDIA):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install timm scikit-learn pandas matplotlib openpyxl onnx onnxruntime
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dữ liệu (không commit): ảnh giải nén vào `images/` ở **thư mục gốc repo** (17.509 file .jpg, MD5 của
`images.zip` = `b7b30f96d466fba86016aa5a26606e0f`), nhãn fold 0 ở `data/labels/` (đã có trong repo).
Nếu để chỗ khác, đặt biến môi trường `LAB_IMAGES` và `LAB_LABELS`.

## Thứ tự chạy

Mọi lệnh chạy từ thư mục này (`submissions/2A202602781_NguyenCongThinh/`). Code tự dùng GPU khi
`torch.cuda.is_available()` (profile `gpu`: 224 px, 12 epoch, batch 64, AMP, đúng công thức nền GUIDE 1.4).

```bash
python code/run_all.py all          # chạy hết; dừng giữa chừng thì chạy lại lệnh này, phần đã xong được bỏ qua
```

hoặc từng giai đoạn:

| # | Lệnh | Việc làm | Sinh ra |
|---|---|---|---|
| 0 | `python code/run_all.py eda` | EDA, kiểm tra split, loss ban đầu ≈ ln 9, overfit 1 batch | `results/eda.json`, `figures/eda_*.png` |
| 1 | `python code/run_all.py backbones` | 6 backbone B01–B06, công thức nền, seed 0 | `runs/B0x/`, `curves/B0x_*.png` |
| 1 | `python code/run_all.py choose_backbone` | độ trễ sơ bộ + chọn backbone (chỉ val) | `results/backbone_latency.csv`, `results/decisions.json` |
| 2 | `python code/run_all.py training` | T00 + 11 ablation T01–T11 (một yếu tố mỗi run) | `runs/T*/`, `curves/T*.png` |
| 2 | `python code/run_all.py choose_combo` → `combo` → `choose_final` | kết hợp yếu tố thắng (T12), chốt công thức | `results/decisions.json` |
| 3 | `python code/run_all.py inference` | TTA, multi-crop, độ phân giải, ensemble, EMA, TS, gộp BN, AMP, ONNX + độ trễ | `results/inference.csv`, `results/latency.csv` |
| 4 | `python code/run_all.py final_train` | F01 seed 0/1/2, T00 seed 1/2 | `runs/F01/`, `runs/T00/` |
| 4 | `python code/run_all.py final_test` | **test một lần mỗi seed**, `eval.py score` + `grade` | `predictions/`, `results/eval/` |
| 5 | `python code/run_all.py results` | bảng tổng hợp | `results.xlsx`, `results/tables.md` |

Kiểm tra code (không cần GPU): `cd code && python -m unittest test_codes -v` (24 test: focal γ=0 ≡ CE,
CutMix λ đúng diện tích, gộp BN chính xác, warmup + cosine, EMA, temperature scaling, ...).

Lưu ý trên Windows:

- `eval.py grade` ghi `grade_I.json` (có tiếng Việt) mà không chỉ định encoding, nên trên Windows lỗi cp1252.
  `final.py` tự gọi eval.py với `PYTHONUTF8=1`; khi tự chạy eval.py bằng tay hãy đặt `set PYTHONUTF8=1`
  (PowerShell: `$env:PYTHONUTF8=1`). Test gốc của repo cũng cần biến này: `PYTHONUTF8=1 python -m unittest discover -s tests`.
- Nếu GPU hết bộ nhớ (8 GB với Swin-T/ConvNeXt-T), giảm batch cho **mọi** thí nghiệm: `set LAB_BATCH=32`
  trước khi chạy, và ghi lại trong báo cáo (GUIDE 1.4).

Lựa chọn giữa các bước được ghi tự động (kèm lý do) trong `results/decisions.json`, theo luật chỉ dùng **val**
(xem docstring của `code/run_all.py`). Muốn chọn khác: ghi khoá vào `results/decisions_override.json`.

## Cấu trúc

```
code/          dataset.py model.py losses.py train.py inference.py benchmark.py   (bộ khung đã hoàn thiện)
               experiments.py run_all.py eda.py infer_experiments.py final.py make_results.py test_codes.py
               lab_day2.ipynb
curves/        một ảnh .png cho mỗi exp_id (B, T, F)
predictions/   <exp_id>_seed<k>_test.csv / _val.csv  (định dạng eval.py)
figures/       EDA, ma trận nhầm lẫn, ảnh lỗi, đánh đổi độ chính xác–độ trễ, reliability diagram
results/       decisions.json, eda.json, inference.csv, latency.csv, eval/ (đầu ra eval.py), tables.md
runs/          checkpoint, logit, history (KHÔNG commit, đã .gitignore)
```
