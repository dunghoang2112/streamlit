# streamlit
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Khởi tạo mô hình AI để nhận diện giới tính
model = tf.keras.models.load_model('model_2.h5')

def process_image(image):
    # Chuyển đổi kích thước ảnh thành 100x100 pixels
    image = image.resize((256,256))
    
    # Chuẩn bị ảnh cho dự đoán
    image = tf.keras.preprocessing.image.img_to_array(image)
    
    image = np.expand_dims(image, axis=0)
    
    # Dự đoán giới tính
    result = model.predict(image)
    
    # Lấy kết quả dự đoán
    gender = 'MALE' if result[0][0] < 0.5 else 'FEMALE'
    
    return gender

def main():
    st.title("Dự đoán giới tính từ ảnh")
    
    uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Đọc file ảnh và hiển thị lên giao diện
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã chọn", use_column_width=True)
        
        # Xử lý ảnh và dự đoán giới tính
        gender = process_image(image)
        
        # Hiển thị kết quả dự đoán
        st.write("Kết quả dự đoán giới tính: ", gender)

if __name__ == "__main__":
    main()
