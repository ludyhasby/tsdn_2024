import streamlit as st
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import ImageEnhance
import pandas as pd
import plotly.express as px



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def decode_label(enc_point):
  switch_dec = {
      0: 'cataract',
      1: 'diabetic_retinopathy',
      2: 'glaucoma',
      3: 'normal'
  }
  return switch_dec.get(enc_point, None) # jika selain range(0, 4) maka default return adalah Null

def predict(model, img, device):
    images = [transform(img.convert("RGB"))]
    images = torch.stack(images).to(device) # to GPU
    model.eval()
    images = images.to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(images).logits
        predictions = torch.argmax(outputs, dim=1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Mengonversi logits menjadi probabilitas

        # dict_result = {}
        disease_list = []
        prob_list = []
        for num, i in enumerate(probabilities): 
           disease_list.append(decode_label(num))
           prob_list.append(i.item()*100)

        result_df = pd.DataFrame({'Disease': disease_list, 'Probability': prob_list}).sort_values(by=['Probability'])

        fig = px.bar(result_df, 
                    x="Probability", 
                    y="Disease", 
                    orientation='h',  # Horizontal
                    title="Barplot Prediksi Klasifikasi Penyakit Mata", 
                    labels={'Disease':'Disease', 'Probability': 'Probability (%)'})

        # Tampilkan plot di Streamlit
        st.plotly_chart(fig)

    return decode_label(predictions[0].item())