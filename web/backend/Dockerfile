# 1. Cài đặt hệ điều hành nền có sẵn Python 3.10 (Bản slim cho nhẹ máy)
FROM python:3.10-slim
#Sửa lỗi docker
RUN apt-get update && apt-get install -y \ 
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo một thư mục làm việc tên là /app ở trong Container
WORKDIR /app


# 3. Copy file danh sách thư viện vào trước để cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ code Python của bạn vào trong /app
COPY . .

# 5. Mở cửa số 8000 để giao tiếp với bên ngoài
EXPOSE 8000

# 6. Lệnh khởi động (Lưu ý: Phải dùng 0.0.0.0 để Docker cho phép kết nối)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
