
# 📌 IMAGE CLASSIFICATION USING CNN ON CIFAR-10

## 📁 Môn học: Nhập môn xử lý ảnh số  
**Trường:** Đại học Văn Lang  
**GVHD:** Thầy Đỗ Hữu Quân, Thầy Nguyễn Thái Anh  
**Sinh viên thực hiện:** Lê Đình Quân – 207CT40568  
**Nhóm:** 25  
**Lớp học phần:** 243_71ITAI40803_01

---

## 📖 Giới thiệu

Dự án này thực hiện xây dựng một hệ thống **phân loại ảnh tự động** sử dụng mạng nơ-ron tích chập (CNN – Convolutional Neural Network) với tập dữ liệu **CIFAR-10**. Đây là đồ án môn học "Nhập môn xử lý ảnh số", kết hợp giữa lý thuyết và thực hành nhằm giúp sinh viên hiểu rõ hơn về:

- Tiền xử lý ảnh
- Thiết kế kiến trúc CNN
- Đánh giá và cải thiện mô hình học sâu

---

## 🧠 Kiến thức sử dụng

- **Xử lý ảnh số**: chuẩn hóa, tăng cường dữ liệu, mã hóa nhãn
- **Mạng CNN**: Conv2D, Pooling, Flatten, Dense, Dropout
- **Huấn luyện mô hình**: optimizer Adam, loss `categorical_crossentropy`
- **Đánh giá mô hình**: accuracy, trực quan hóa loss/accuracy theo epoch

---

## 🛠️ Công nghệ

- 🔤 **Ngôn ngữ**: Python 3.10  
- 📦 **Thư viện chính**:
  - TensorFlow, Keras
  - NumPy, Matplotlib, scikit-learn  
- 🧪 **Nền tảng chạy mô hình**: Google Colab (hỗ trợ GPU)

---

## 📂 Cấu trúc mô hình CNN

```python
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

---

## 🧪 Kết quả thực nghiệm

- 🎯 Độ chính xác đạt ~**80%** trên tập kiểm tra sau 10 epoch
- 📉 Loss giảm đều, accuracy tăng ổn định
- ✅ Không overfitting rõ rệt
- 🆚 So với random baseline (10%), mô hình vượt trội

---

## 📊 Dataset: CIFAR-10

- 📷 60.000 ảnh màu RGB (32x32)
- 🧩 10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🚀 Hướng phát triển

- Nâng cao mô hình với ResNet, VGG, hoặc Transfer Learning
- Áp dụng kỹ thuật tăng cường dữ liệu (Data Augmentation)
- Triển khai giao diện cho người dùng upload ảnh và phân loại
- So sánh nhiều kiến trúc CNN khác nhau

---

## 🔗 Liên kết

- 📁 Notebook mô hình: [Doancuoiky.ipynb](https://github.com/Ldinhquan/-n-cu-i-k---Nh-p-m-n-x-l-nh-s-/blob/main/Doancuoiky.ipynb)

---

## 📚 Tài liệu tham khảo

- CIFAR-10 Dataset: Krizhevsky & Hinton (2009)
- Keras Examples: [Image classification with modern convnets](https://keras.io/examples/vision/image_classification_from_scratch/)
- Kaggle CIFAR-10 CNN: https://www.kaggle.com/code/faressayah/cifar-10-images-classification-using-cnns-88
- Papers With Code: [Image Classification on CIFAR-10](https://paperswithcode.com/sota/image-classification-on-cifar-10)
