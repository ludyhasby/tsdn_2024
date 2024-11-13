import streamlit as st
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from models import load_model
from predictions import predict
import torch
import io
from maps import saliency_map
from torchvision import transforms

st.set_page_config(
    page_title="PANDAWA - Pendeteksi Awal Penyakit Mata",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
st.write("Panduan Penggunaan")
c1, c2 = st.columns((3, 1))
with c1: 
    st.markdown("""
    - Klik "Browse Files" untuk mengunggah gambar dengan format gambar JPG, PNG, atau JPEG seperti contoh di samping.
    - Pilih gambar fundus dari perangkat anda.
    - Setelah itu, sistem akan memuat gambar dan secara otomatis mendeteksi penyakit yang ada pada fundus dengan fitur peningkatan kualitas gambar dengan faktor 3. 
    - Anda dapat meningkatkan nilai faktor, jika kualitas gambar dirasa kurang. 
    """)
with c2: 
    st.image('contoh_fundus.png')
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
    result, pred = predict(test_model, enhanced_image, device, transform)
    st.write(f"Hasil Prediksi : {result}")
    
    saliency_map(test_model, enhanced_image, device, transform, pred)
