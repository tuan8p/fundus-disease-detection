import os
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
from functools import partial
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import apply_preprocessing, list_strategies, validate_strategy

# PIL tăng giới hạn ảnh lớn
Image.MAX_IMAGE_PIXELS = None


def process_image(img_path: str, out_dir: str, strategy: str) -> tuple[bool, str]:
    """
    Xử lý một ảnh: đọc → preprocess → lưu.
    Trả về (success, img_path) để dễ track lỗi.
    """
    try:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, img_name)

        if os.path.exists(out_path):
            return True, img_path

        img = Image.open(img_path).convert("RGB")
        img_processed = apply_preprocessing(img, strategy)

        # Lưu PNG với compress_level=1 (nhanh hơn mặc định=6, file lớn hơn chút)
        img_processed.save(out_path, format="PNG", compress_level=1)
        del img
        del img_processed
        return True, img_path

    except Exception as e:
        return False, f"{img_path} — Lỗi: {e}"


def process_batch(
    image_paths: list[str],
    out_dir: str,
    strategy: str,
    num_workers: int,
    split_name: str,
) -> int:
    """
    Xử lý một batch ảnh bằng ThreadPoolExecutor.
    Trả về số ảnh lỗi.
    """
    if not image_paths:
        return 0

    print(f"\nĐang xử lý [{split_name}] — {len(image_paths)} ảnh, {num_workers} threads...")

    errors = 0
    fn = partial(process_image, out_dir=out_dir, strategy=strategy)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fn, p): p for p in image_paths}

        with tqdm(total=len(futures), unit="img", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                success, info = future.result()
                if not success:
                    errors += 1
                    tqdm.write(f"  ✗ {info}")
                pbar.update(1)

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý offline dataset APTOS 2019 — tối ưu cho Kaggle CPU"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset gốc (chứa train.csv, train_images/, test_images/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Thư mục lưu dataset sau xử lý")
    parser.add_argument("--strategy", type=str, required=True,
                        help=f"Strategy xử lý. Chọn: {list_strategies()}")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Số threads (mặc định 4 — phù hợp Kaggle 2-core CPU)")
    parser.add_argument("--splits", type=str, default="train,test",
                        help="Splits cần xử lý, cách nhau bởi dấu phẩy (mặc định: train,test)")

    args = parser.parse_args()
    validate_strategy(args.strategy)

    splits = [s.strip() for s in args.splits.split(",")]

    # ── Tạo thư mục output ───────────────────────────────────────────────
    split_dirs = {}
    for split in splits:
        d = os.path.join(args.output_dir, f"{split}_images")
        os.makedirs(d, exist_ok=True)
        split_dirs[split] = d

    # ── Copy CSV ─────────────────────────────────────────────────────────
    for csv_file in ["train.csv", "test.csv", "sample_submission.csv"]:
        src = os.path.join(args.data_dir, csv_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, csv_file))

    # ── Thu thập ảnh ─────────────────────────────────────────────────────
    split_images = {}
    for split in splits:
        paths = glob(os.path.join(args.data_dir, f"{split}_images", "*.png"))
        split_images[split] = paths

    # ── Header ───────────────────────────────────────────────────────────
    total_imgs = sum(len(v) for v in split_images.values())
    print("=" * 60)
    print("  TIỀN XỬ LÝ OFFLINE — APTOS 2019")
    print("=" * 60)
    print(f"  Strategy  : {args.strategy}")
    print(f"  Workers   : {args.num_workers} threads")
    print(f"  Nguồn     : {args.data_dir}")
    print(f"  Đích      : {args.output_dir}")
    for split in splits:
        print(f"  {split:8s}  : {len(split_images[split])} ảnh")
    print(f"  Tổng      : {total_imgs} ảnh")
    print("-" * 60)

    if args.strategy == "none":
        print("⚠  Strategy 'none': chỉ copy ảnh, không xử lý.")

    # ── Xử lý từng split ────────────────────────────────────────────────
    total_errors = 0
    for split in splits:
        errors = process_batch(
            image_paths=split_images[split],
            out_dir=split_dirs[split],
            strategy=args.strategy,
            num_workers=args.num_workers,
            split_name=f"{split}_images",
        )
        total_errors += errors

    # ── Footer ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if total_errors == 0:
        print(f"✓ HOÀN TẤT! Lưu tại: {os.path.abspath(args.output_dir)}")
    else:
        print(f"⚠ HOÀN TẤT với {total_errors} lỗi. Kiểm tra log phía trên.")
    print("=" * 60)
    print("BƯỚC TIẾP THEO:")
    print(f"  DATA_DIR = '{os.path.abspath(args.output_dir)}'")
    print("  PREPROCESSING_STRATEGY = 'none'  (ảnh đã xử lý sẵn)")


if __name__ == "__main__":
    main()