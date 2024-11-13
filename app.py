import streamlit as st
from PIL import Image 
import matplotlib.pyplot as plt
from models import load_model
from predictions import predict
import torch
import io
from PIL import ImageEnhance

def load_image(file_upload):
    if file_upload is not None:
        image_data = file_upload.getvalue()
        return Image.open(io.BytesIO(image_data))
    else:
        return None    


# device cuda, model path loader 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'efficientnet_model.pth'

st.title("Klasifikasi Penyakit Mata")
uploaded_file = st.file_uploader("Pilih Image", type=['jpg', 'png', 'jpeg'])
if uploaded_file: 
    st.write("Tingkatkan kualitas gambar dengan Image Enhancement")
    factor = st.number_input("Masukkan faktor peningkatan gambar", min_value=1, max_value=10, value=3)
    original_image = load_image(uploaded_file)
    col1, col2 = st.columns(2)
    with col1: 
        st.write("Before Enhancement")
        st.image(original_image)
    with col2: 
        st.write("After Enhancement")
        enhanced_image = ImageEnhance.Sharpness(original_image).enhance(factor)
        st.image(enhanced_image)

    test_model = load_model(model_path, device)
    result = predict(test_model, enhanced_image, device)
    st.write(f"Hasil Prediksi : {result}")