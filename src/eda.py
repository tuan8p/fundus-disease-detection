import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from matplotlib.backends.backend_pdf import PdfPages # Thư viện xuất PDF

CSV_PATH = r'D:\BTL_DL\data\train.csv'
IMAGE_DIR = r'D:\BTL_DL\data\train_images/' 
OUTPUT_CSV = 'image_eda_features.csv'


sns.set_theme(style="whitegrid")


def extract_image_features(image_id):
    img_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    
    features = {
        'id_code': image_id,
        'width': np.nan, 'height': np.nan, 'aspect_ratio': np.nan,
        'brightness': np.nan, 'blur_score': np.nan,
        'mean_r': np.nan, 'mean_g': np.nan, 'mean_b': np.nan,
        'black_border_ratio': np.nan
    }
    
    if not os.path.exists(img_path):
        return features
        
    img = cv2.imread(img_path)
    if img is None:
        return features
        
    try:
        h, w, _ = img.shape
        features['width'] = w
        features['height'] = h
        features['aspect_ratio'] = round(w / h, 3)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        features['brightness'] = round(np.mean(gray), 2)
        features['blur_score'] = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)
        
        features['mean_r'] = round(np.mean(img_rgb[:,:,0]), 2)
        features['mean_g'] = round(np.mean(img_rgb[:,:,1]), 2)
        features['mean_b'] = round(np.mean(img_rgb[:,:,2]), 2)
        
        black_pixels = np.sum(gray < 10)
        total_pixels = w * h
        features['black_border_ratio'] = round(black_pixels / total_pixels, 3)
        
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_id}: {e}")
        
    return features

def run_extraction_pipeline():
    print("1. Đọc file Nhãn (train.csv)...")
    train_df = pd.read_csv(CSV_PATH)
    image_ids = train_df['id_code'].tolist()
    
    print(f"2. Bắt đầu trích xuất đặc trưng cho {len(image_ids)} ảnh...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(extract_image_features, image_ids), total=len(image_ids)):
            results.append(res)
            
    features_df = pd.DataFrame(results)
    final_df = pd.merge(train_df, features_df, on='id_code', how='left')
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Đã lưu kết quả vào: {OUTPUT_CSV}")
    return final_df

def plot_and_export_eda(df, output_pdf="EDA_Report_VongMac.pdf", output_img_dir="EDA_Images"):
    print(f"\n--- BẮT ĐẦU VẼ VÀ XUẤT BÁO CÁO EDA ---")
    df = df.dropna().copy()
    
    os.makedirs(output_img_dir, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        
        print("Đang vẽ Trang 1: Đặc tính quang học...")
        fig1 = plt.figure(figsize=(20, 16))
        
        plt.subplot(3, 2, 1)
        sns.countplot(data=df, x='diagnosis', palette='viridis')
        plt.title('Phân phối Cấp độ Bệnh (Class Imbalance)', fontsize=14, pad=10)
        
        plt.subplot(3, 2, 2)
        sns.histplot(df['aspect_ratio'], bins=30, kde=True, color='coral')
        plt.title('Phân phối Tỷ lệ Khung hình (Width / Height)', fontsize=14, pad=10)
        plt.axvline(1.0, color='red', linestyle='--')
        
        plt.subplot(3, 2, 3)
        sns.scatterplot(data=df, x='brightness', y='blur_score', hue='diagnosis', palette='tab10', alpha=0.6)
        plt.title('Độ Sáng vs Độ Nét', fontsize=14, pad=10)
        plt.yscale('log')
        
        plt.subplot(3, 2, 4)
        sns.boxplot(data=df, x='diagnosis', y='black_border_ratio', palette='Set2')
        plt.title('Tỷ lệ Viền đen trên từng cấp độ', fontsize=14, pad=10)
        
        plt.subplot(3, 2, 5)
        sns.kdeplot(df['mean_r'], color='red', fill=True, alpha=0.3)
        sns.kdeplot(df['mean_g'], color='green', fill=True, alpha=0.3)
        sns.kdeplot(df['mean_b'], color='blue', fill=True, alpha=0.3)
        plt.title('Phân phối Màu sắc Trung bình', fontsize=14, pad=10)
        
        plt.subplot(3, 2, 6)
        corr_cols = ['diagnosis', 'aspect_ratio', 'brightness', 'blur_score', 'mean_r', 'mean_g', 'mean_b', 'black_border_ratio']
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Ma trận Tương quan', fontsize=14, pad=10)
        
        plt.tight_layout(pad=3.0, h_pad=5.0, w_pad=4.0)
        
        pdf.savefig(fig1) 
        fig1.savefig(os.path.join(output_img_dir, "01_Optical_Features.png"), dpi=150, bbox_inches='tight')
        plt.close(fig1) 


        print("Đang vẽ Trang 2: Thống kê kích thước...")
        per_class = df.groupby('diagnosis').agg(
            min_width=('width', 'min'), max_width=('width', 'max'),
            min_height=('height', 'min'), max_height=('height', 'max'),
            mean_width=('width', 'mean'), mean_height=('height', 'mean'),
        ).reset_index()

        sns.set_palette("husl")
        fig2 = plt.figure(figsize=(20, 6))
        colors = sns.color_palette("husl", len(per_class))

        ax1 = plt.subplot(1, 3, 1)
        x = np.arange(len(per_class))
        ax1.bar(x - 0.175, per_class['mean_width'], 0.35, label='Mean Width', alpha=0.8)
        ax1.bar(x + 0.175, per_class['mean_height'], 0.35, label='Mean Height', alpha=0.8)
        ax1.set_title("Trung bình Kích thước", fontsize=14, pad=15)
        ax1.set_xticks(x); ax1.set_xticklabels(per_class['diagnosis'])
        ax1.legend()

        ax2 = plt.subplot(1, 3, 2)
        for i, row in per_class.iterrows():
            ax2.plot([row['min_width'], row['max_width']], [i, i], 'o-', linewidth=3, markersize=8, color=colors[i])
            ax2.text(row['mean_width'], i, f"  {row['mean_width']:.0f}", va='center', fontweight='bold')
        ax2.set_yticks(range(len(per_class))); ax2.set_yticklabels(per_class['diagnosis'])
        ax2.set_title("Dải biến thiên Chiều Rộng", fontsize=14, pad=15)

        ax3 = plt.subplot(1, 3, 3)
        for i, row in per_class.iterrows():
            ax3.plot([row['min_height'], row['max_height']], [i, i], 's-', linewidth=3, markersize=8, color=colors[i])
            ax3.text(row['mean_height'], i, f"  {row['mean_height']:.0f}", va='center', fontweight='bold')
        ax3.set_yticks(range(len(per_class))); ax3.set_yticklabels(per_class['diagnosis'])
        ax3.set_title("Dải biến thiên Chiều Cao", fontsize=14, pad=15)

        plt.suptitle("Thống kê Phân phối Kích thước Ảnh", fontsize=18, fontweight='bold')
        plt.tight_layout(pad=3.0, w_pad=5.0) 
        plt.subplots_adjust(top=0.85)
        
        pdf.savefig(fig2) 
        fig2.savefig(os.path.join(output_img_dir, "02_Dimension_Stats.png"), dpi=150, bbox_inches='tight')
        plt.close(fig2)

    print(f"\nHOÀN TẤT! File của bạn nằm ở:")
    print(f" Bản PDF (gộp): {output_pdf}")
    print(f"Ảnh rời (PNG): Thư mục '{output_img_dir}/'")


if __name__ == '__main__':
    df_analyzed = run_extraction_pipeline()