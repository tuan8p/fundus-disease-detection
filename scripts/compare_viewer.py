import streamlit as st
import os
from PIL import Image
import glob

def main():
    st.set_page_config(layout="wide", page_title="APTOS Preprocessing Viewer")
    
    st.title("👁️ APTOS 2019 Preprocessing Comparison")
    st.markdown("Ứng dụng web hiển thị các phiên bản tiền xử lý của cùng một ảnh.")

    # Cấu hình danh sách các thư mục chứa ảnh
    # Lưu ý: Thêm "train_images" hoặc tuỳ chỉnh theo cấu trúc thư mục thực tế của bạn
    RAW_DIR = "/home/minhtriet/Downloads/aptos2019-blindness-detection/train_images"
    ROI_DIR = "/home/minhtriet/Documents/fundus-disease-detection/aptos_roi/train_images"
    BEN_DIR = "/home/minhtriet/Documents/fundus-disease-detection/aptos_ben/train_images"
    CLAHE_DIR = "/home/minhtriet/Documents/fundus-disease-detection/aptos_clahe/train_images"

    # Lấy danh sách ảnh từ thư mục Raw (hoặc bất kỳ thư mục nào đã được xử lý)
    if os.path.exists(RAW_DIR):
        # Lấy file .png (hoặc thay đổi đuôi tuỳ ý)
        sample_images = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    elif os.path.exists(BEN_DIR):
        sample_images = sorted([f for f in os.listdir(BEN_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    else:
        st.error(f"Vui lòng kiểm tra lại đường dẫn. Không tìm thấy {RAW_DIR} hoặc {BEN_DIR}")
        return
        
    if not sample_images:
        st.warning("Không tìm thấy ảnh PNG/JPG nào trong thư mục.")
        return

    # Giao diện chọn ảnh
    col_nav, col_info = st.columns([1, 2])
    with col_nav:
        selected_img = st.selectbox("📌 Chọn Sample ID (tên ảnh):", sample_images)
    
    with col_info:
        st.info("Sử dụng nút mũi tên trên bàn phím trong hộp Dropdown để lướt qua nhiều ảnh một cách nhanh chóng.")

    # Giao diện hiển thị: 2 dòng x 2 cột
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    def show_img(col, title, path):
        with col:
            st.markdown(f"### {title}")
            if os.path.exists(path):
                img = Image.open(path)
                st.image(img, width=400)
                # st.caption(f"`{path}`")
            else:
                st.error(f"❌ Không tìm thấy file: {path}")
                
    st.divider()

    # Dòng 1
    show_img(col1, "1. Gốc (Raw)", os.path.join(RAW_DIR, selected_img))
    show_img(col2, "2. ROI (aptos_roi)", os.path.join(ROI_DIR, selected_img))
    show_img(col3, "3. CLAHE (aptos_clahe)", os.path.join(CLAHE_DIR, selected_img))    
    show_img(col4, "4. Ben (aptos_ben)", os.path.join(BEN_DIR, selected_img))


if __name__ == "__main__":
    main()
