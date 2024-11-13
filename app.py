import streamlit as st
from PIL import Image 
import matplotlib.pyplot as plt
from models import load_model
from predictions import predict
import torch
import io

def load_image(file_upload):
    if file_upload is not None:
        image_data = file_upload.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None    


# device cuda, model path loader 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'efficientnet_model.pth'

st.title("Klasifikasi Penyakit Mata")
upluoded_file = st.file_uploader("Pilih Image", type=['jpg', 'png', 'jpeg'])
if upluoded_file: 
    image = load_image(upluoded_file)

    test_model = load_model(model_path, device)
    result = predict(test_model, image, device)
    st.write(f"Hasil Prediksi : {result}")