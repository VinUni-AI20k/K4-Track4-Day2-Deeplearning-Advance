"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# [Implemented by Claude (AI assistant)] extra imports needed by the implementation below.
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)

# [Implemented by Claude (AI assistant)] constants used by check_split.
EXPECTED_TOTAL = 17509                     # số ảnh của DeepWeeds (Table 1 của bài báo)
EXPECTED_RATIO = {"train": 0.6, "val": 0.2, "test": 0.2}
RATIO_TOLERANCE = 0.01                     # lệch hơn ~1 điểm phần trăm thì cảnh báo (README 2.1)


def eval_resize(img_size: int) -> int:
    """[Implemented by Claude (AI assistant)] Kích thước resize trước khi center-crop lúc đánh giá.

    Input : img_size (int) - kích thước crop cuối cùng (ví dụ 224 hoặc 128).
    Output: int - cạnh ảnh sau resize, giữ tỉ lệ 256/224 của công thức nền
            (224 -> 256 tức là giữ nguyên ảnh gốc 256x256; 128 -> 146).
    """
    return int(round(img_size * 256 / 224))


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    # [Implemented by Claude (AI assistant)]
    # Input : labels_dir (str | Path) - thư mục chứa train/val/test_subset{fold}.csv
    #         fold (int)              - số fold (bài lab bắt buộc fold 0, quy tắc S1)
    # Output: tuple (train_df, val_df, test_df); mỗi phần tử là pandas.DataFrame có ít nhất
    #         cột `Filename` (str) và `Label` (int 0..8), đúng thứ tự dòng của file gốc.
    # Cách làm: đọc nguyên văn 3 file CSV (không lọc, không chia lại), chỉ kiểm tra có đủ cột
    #         và ép `Label` về int. Mọi hàm khác (check_split, make_loader, train.run) dùng kết quả này.
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        df = pd.read_csv(path)
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{path}: cần cột Filename và Label, nhận {list(df.columns)}")
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    # [Implemented by Claude (AI assistant)]
    # Input : train_df, val_df, test_df (DataFrame từ load_split); images_dir (str | Path) thư mục ảnh .jpg
    # Output: dict cấu trúc như sau (mọi số là int/float thuần để ghi được ra JSON):
    #   {
    #     "n":         {"train": 10501, "val": 3501, "test": 3507, "total": 17509},
    #     "ratio":     {"train": 0.5998, "val": 0.2000, "test": 0.2003},
    #     "per_class": {"train": {"Chinee Apple": 675, ...}, "val": {...}, "test": {...},
    #                   "all": {"Chinee Apple": 1125, ...}},
    #     "overlap":   {"train&val": 0, "train&test": 0, "val&test": 0},
    #     "duplicates_within": {"train": 0, "val": 0, "test": 0},
    #     "union": 17509, "missing_files": 0, "ratio_warning": False
    #   }
    # Cách làm: đếm số ảnh từng tập/lớp, tính giao theo tên file giữa từng cặp tập, hợp ba tập,
    #   và so tên file với danh sách file thật trong images_dir. Vi phạm (giao khác rỗng, hợp khác
    #   17.509, thiếu file, trùng tên trong một tập) -> AssertionError để dừng ngay. Lệch tỉ lệ
    #   60/20/20 quá 1 điểm % chỉ in cảnh báo (README yêu cầu báo giảng viên, không tự sửa).
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(df["Filename"]) for k, df in splits.items()}
    total = sum(len(df) for df in splits.values())

    dup = {k: int(df["Filename"].duplicated().sum()) for k, df in splits.items()}
    overlap = {"train&val": len(names["train"] & names["val"]),
               "train&test": len(names["train"] & names["test"]),
               "val&test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    on_disk = set(os.listdir(images_dir))
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)

    per_class = {}
    for k, df in splits.items():
        counts = df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        per_class[k] = {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)}
    per_class["all"] = {c: sum(per_class[k][c] for k in splits) for c in CLASS_NAMES}

    ratio = {k: len(df) / total for k, df in splits.items()}
    ratio_warning = any(abs(ratio[k] - EXPECTED_RATIO[k]) > RATIO_TOLERANCE for k in splits)

    report = {
        "n": {**{k: int(len(df)) for k, df in splits.items()}, "total": int(total)},
        "ratio": {k: round(v, 4) for k, v in ratio.items()},
        "per_class": per_class,
        "overlap": overlap,
        "duplicates_within": dup,
        "union": int(union),
        "missing_files": len(missing),
        "ratio_warning": bool(ratio_warning),
    }

    print(f"[check_split] n = {report['n']}  ratio = {report['ratio']}")
    print(f"[check_split] overlap = {overlap}  union = {union}  missing files = {len(missing)}")
    if ratio_warning:
        print("[check_split] CẢNH BÁO: tỉ lệ lệch 60/20/20 hơn 1 điểm %, cần báo giảng viên.")

    assert all(v == 0 for v in dup.values()), f"có tên file trùng trong một tập: {dup}"
    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    assert union == EXPECTED_TOTAL, f"hợp ba tập = {union}, kỳ vọng {EXPECTED_TOTAL}"
    assert not missing, f"{len(missing)} file không có trong {images_dir}, ví dụ {missing[:5]}"
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    # [Implemented by Claude (AI assistant)]
    # Input : train (bool) - True: transform huấn luyện (ngẫu nhiên); False: transform đánh giá (tất định)
    #         img_size (int) - cạnh ảnh đưa vào model
    #         aug (str) - mức augmentation khi train:
    #             "basic"   : RandomResizedCrop + lật ngang                     (công thức nền T00)
    #             "flipv"   : basic + lật dọc (ảnh chụp từ trên xuống, lật dọc vẫn là cùng loài cây)
    #             "color"   : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
    #             "trivial" : basic + TrivialAugmentWide
    #             "randaug" : basic + RandAugment(num_ops=2, magnitude=9)
    # Output: torchvision.transforms.v2.Compose nhận ảnh PIL HOẶC tensor uint8 (3, H, W) và trả về
    #         tensor float32 (3, img_size, img_size) đã chuẩn hoá theo mean/std ImageNet.
    # Cách làm: v2.ToImage() đưa mọi đầu vào về tv_tensors.Image -> các phép hình học/màu trên uint8
    #   -> ToDtype(float32, scale=True) -> Normalize. Đánh giá: Resize(eval_resize(img_size)) rồi
    #   CenterCrop(img_size) (với 224 nghĩa là crop 224 giữa ảnh gốc 256). Không có phép ngẫu nhiên nào.
    #   Mixup/CutMix không nằm ở đây (losses.mix_batch làm theo batch).
    finish = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return v2.Compose([v2.ToImage(), v2.Resize(eval_resize(img_size), antialias=True),
                           v2.CenterCrop(img_size), *finish])

    extra = {
        "basic": [],
        "flipv": [v2.RandomVerticalFlip()],
        "color": [v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)],
        "trivial": [v2.TrivialAugmentWide()],
        "randaug": [v2.RandAugment(num_ops=2, magnitude=9)],
    }
    if aug not in extra:
        raise ValueError(f"aug không hợp lệ: {aug!r}; chọn một trong {sorted(extra)}")
    return v2.Compose([v2.ToImage(), v2.RandomResizedCrop(img_size, antialias=True),
                       v2.RandomHorizontalFlip(), *extra[aug], *finish])


def preload_images(images_dir: str | Path, size: int, cache_dir: str | Path) -> tuple[np.ndarray, dict]:
    """[Implemented by Claude (AI assistant)] Giải mã MỘT lần toàn bộ ảnh và lưu bộ nhớ đệm .npy.

    Input : images_dir (str | Path) - thư mục 17.509 ảnh .jpg 256x256
            size (int)              - cạnh ảnh lưu trong bộ đệm (ảnh vuông size x size)
            cache_dir (str | Path)  - nơi ghi images_<size>.npy và images_<size>_names.txt
    Output: (array, index)
            array : np.memmap uint8 dạng (N_all, 3, size, size), chỉ đọc
            index : dict {filename (str): vị trí dòng (int)} để tra ảnh theo tên file
    Cách làm: lần đầu giải mã song song bằng ThreadPool (PIL nhả GIL khi decode), resize về size x size
      (ảnh DeepWeeds đều vuông 256x256 nên không méo), ghi .npy. Những lần sau chỉ mở memmap,
      nên train/val/test và các tiến trình worker dùng chung một bản trên đĩa/page cache.
      Mục đích: trên máy chỉ có CPU, giải mã JPEG mỗi epoch là nút cổ chai (GUIDE mục 7: nạp trước ảnh).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path = cache_dir / f"images_{size}.npy"
    names_path = cache_dir / f"images_{size}_names.txt"
    if not (arr_path.exists() and names_path.exists()):
        names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

        def _load(name):
            with Image.open(Path(images_dir) / name) as im:
                im = im.convert("RGB")
                if im.size != (size, size):
                    im = im.resize((size, size), Image.BILINEAR, reducing_gap=None)
                return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)

        tmp_path = cache_dir / f"images_{size}.tmp.npy"
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(len(names), 3, size, size))
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            for i, a in enumerate(ex.map(_load, names, chunksize=64)):
                out[i] = a
        out.flush()
        del out
        os.replace(tmp_path, arr_path)
        names_path.write_text("\n".join(names), encoding="utf-8")
        print(f"[preload_images] đã tạo bộ đệm {arr_path} ({len(names)} ảnh, {size}x{size})")
    names = names_path.read_text(encoding="utf-8").split("\n")
    array = np.load(arr_path, mmap_mode="r")
    return array, {n: i for i, n in enumerate(names)}


def seed_worker(worker_id: int) -> None:
    """[Implemented by Claude (AI assistant)] worker_init_fn cho DataLoader.

    Input : worker_id (int) - do DataLoader truyền vào. Output: None.
    Cách làm: mỗi worker nhận seed torch khác nhau (torch.initial_seed() đã được DataLoader đặt từ
      generator của loader); ta suy ra seed cho numpy và random từ đó để augmentation tái lập được.
    """
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


class DeepWeedsDataset(Dataset):  # TODO: kế thừa torch.utils.data.Dataset
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 preload_size: int | None = None, cache_dir: str | Path | None = None):
        # [Implemented by Claude (AI assistant)]
        # Input : df (DataFrame có Filename, Label), images_dir (thư mục ảnh), transform (callable hoặc None),
        #         preload_size (int | None) - nếu có: đọc ảnh từ bộ đệm uint8 size x size (preload_images)
        #         thay vì giải mã JPEG mỗi lần; cache_dir - nơi đặt bộ đệm (mặc định runs/cache, thư mục bị .gitignore).
        # Output: đối tượng Dataset; giữ nguyên thứ tự dòng của df (quan trọng để ghép logit với tên file).
        # Cách làm: lưu danh sách tên file + nhãn; bộ đệm memmap được mở "lười" (lần đầu __getitem__)
        #   để khi DataLoader nhân bản dataset sang worker không phải pickle cả mảng ảnh.
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).to_numpy()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.preload_size = preload_size
        self.cache_dir = Path(cache_dir) if cache_dir else Path("runs") / "cache"
        self._array = None
        self._rows = None
        if preload_size:
            _, index = preload_images(self.images_dir, preload_size, self.cache_dir)  # tạo bộ đệm nếu chưa có
            self._rows = np.array([index[f] for f in self.filenames], dtype=np.int64)

    def __len__(self) -> int:
        # [Implemented by Claude (AI assistant)] Output: số ảnh (int) = số dòng của df.
        return len(self.filenames)

    def __getitem__(self, i: int):
        # [Implemented by Claude (AI assistant)]
        # Input : i (int) - chỉ số ảnh trong df
        # Output: (image, label, filename) với image = tensor float (3, img_size, img_size) sau transform
        #         (hoặc PIL / tensor uint8 nếu transform=None), label = int 0..8, filename = str.
        # Cách làm: lấy ảnh từ bộ đệm memmap (tensor uint8 3xSxS) nếu preload, ngược lại mở bằng PIL và
        #   chuyển RGB; rồi áp transform.
        name = self.filenames[i]
        if self.preload_size:
            if self._array is None:
                self._array, _ = preload_images(self.images_dir, self.preload_size, self.cache_dir)
            img = torch.from_numpy(np.array(self._array[self._rows[i]]))
        else:
            with Image.open(self.images_dir / name) as im:
                img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), name


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                preload_size: int | None = None, cache_dir: str | Path | None = None,
                seed: int = 0, drop_last: bool | None = None):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    # [Implemented by Claude (AI assistant)]
    # Input : df, images_dir, transform - như DeepWeedsDataset; batch_size (int);
    #         train (bool) - True: xáo trộn + drop_last; False: giữ đúng thứ tự df, không bỏ batch nào
    #         sampler (None | "balanced") - "balanced": WeightedRandomSampler trọng số 1/n_lớp (trục D)
    #         num_workers (int); preload_size, cache_dir - chuyển cho DeepWeedsDataset;
    #         seed (int) - seed của generator (thứ tự batch + seed worker); drop_last (bool | None) - None = train
    # Output: torch.utils.data.DataLoader trả về batch (images[B,3,H,W], labels[B] (tensor int64), filenames (list[str]))
    # Cách làm: dựng dataset; train -> shuffle bằng generator đã seed (hoặc sampler cân bằng, có hoàn lại,
    #   số mẫu mỗi epoch = len(df)); eval -> shuffle=False. pin_memory chỉ bật khi có CUDA;
    #   persistent_workers để không tạo lại tiến trình mỗi epoch (trên Windows tạo worker rất chậm).
    ds = DeepWeedsDataset(df, images_dir, transform, preload_size=preload_size, cache_dir=cache_dir)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    shuffle = train
    if sampler == "balanced":
        if not train:
            raise ValueError("sampler cân bằng chỉ dùng khi train")
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES).astype(np.float64)
        weights = 1.0 / counts[ds.labels]
        smp = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
        shuffle = False
    elif sampler not in (None, "none"):
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=smp,
        drop_last=train if drop_last is None else drop_last,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=num_workers > 0,
    )
