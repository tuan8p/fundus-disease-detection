import os
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import shutil
import sys

# Đưa src vào hệ thống để import thư viện của bạn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import apply_preprocessing, list_strategies, validate_strategy

def process_image(img_path, out_dir, strategy):
    """
    Hàm xử lý một ảnh: đọc ảnh, chạy tiền xử lý và lưu xuống ổ cứng.
    """
    try:
        # Giữ nguyên tên file (VD: id_code.png)
        img_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, img_name)
        
        # Nếu file đã tồn tại thì bỏ qua (hỗ trợ resume nếu bị ngắt ngang)
        if os.path.exists(out_path):
            return True
            
        img = Image.open(img_path).convert("RGB")
        img_processed = apply_preprocessing(img, strategy)
        img_processed.save(out_path, format="PNG")
        return True
    except Exception as e:
        print(f"\nLỗi khi xử lý ảnh {img_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Script tiền xử lý toàn bộ dataset APTOS 2019 offline")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Đường dẫn đến dataset gốc (chứa train.csv, train_images/, test_images/)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Đường dẫn lưu dataset sau khi xử lý (ví dụ: data/aptos_roi_ben_clahe/)")
    parser.add_argument("--strategy", type=str, required=True, 
                        help=f"Chiến lược xử lý (các lựa chọn: {list_strategies()})")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), 
                        help="Số luồng CPU/Worker để tăng tốc độ chạy (mặc định lấy tối đa CPU)")
    
    args = parser.parse_args()
    
    # Kiểm tra tính hợp lệ của strategy
    validate_strategy(args.strategy)
    if args.strategy == "none":
        print("Cảnh báo: Strategy là 'none', script sẽ chỉ copy ảnh gốc mà không qua xử lý gì.")
    
    # 1. Tạo câu trúc thư mục output
    train_out = os.path.join(args.output_dir, "train_images")
    test_out = os.path.join(args.output_dir, "test_images")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)
    
    # 2. Copy luôn các file csv qua output_dir để thư mục output có thể thay thế thẳng cho data_dir cũ
    for csv_file in ["train.csv", "test.csv", "sample_submission.csv"]:
        src_csv = os.path.join(args.data_dir, csv_file)
        if os.path.exists(src_csv):
            shutil.copy(src_csv, os.path.join(args.output_dir, csv_file))
            
    # 3. Thu thập danh sách ảnh
    train_images = glob(os.path.join(args.data_dir, "train_images", "*.png"))
    test_images = glob(os.path.join(args.data_dir, "test_images", "*.png"))
    
    print("=" * 60)
    print(" BẮT ĐẦU TIỀN XỬ LÝ OFFLINE ")
    print("=" * 60)
    print(f"Strategy   : {args.strategy}")
    print(f"Workers    : {args.num_workers}")
    print(f"Nguồn      : {args.data_dir}")
    print(f"Đích       : {args.output_dir}")
    print(f"Train. imgs: {len(train_images)}")
    print(f"Test. imgs : {len(test_images)}")
    print("-" * 60)
    
    # 4. Xử lý đa luồng (Multiprocessing) để tận dụng hết CPU
    if train_images:
        print("\nĐang xử lý [train_images]...")
        process_train = partial(process_image, out_dir=train_out, strategy=args.strategy)
        with mp.Pool(args.num_workers) as pool:
            # dùng tqdm để hiển thị thanh tiến trình
            list(tqdm(pool.imap(process_train, train_images), total=len(train_images)))
            
    if test_images:
        print("\nĐang xử lý [test_images]...")
        process_test = partial(process_image, out_dir=test_out, strategy=args.strategy)
        with mp.Pool(args.num_workers) as pool:
            list(tqdm(pool.imap(process_test, test_images), total=len(test_images)))
            
    print("\n" + "=" * 60)
    print(f"HOÀN TẤT! Dataset mới đã được lưu tại: {os.path.abspath(args.output_dir)}")
    print("=" * 60)
    print("HƯỚNG DẪN SỬ DỤNG CHO TRAINING:")
    print(" 1. Mở file configs hoặc notebook (vd: baseline.ipynb)")
    print(f" 2. Đổi DATA_DIR = '{os.path.abspath(args.output_dir)}'")
    print(" 3. Nhớ đổi PREPROCESSING_STRATEGY = 'none' (vì ảnh đã được xử lý sẵn)")

if __name__ == "__main__":
    main()
