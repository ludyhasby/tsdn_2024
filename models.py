import streamlit as st
from transformers import AutoModelForImageClassification
import torch

@st.cache_resource
def load_model(model_path, device):
    model = AutoModelForImageClassification.from_pretrained("efficientnet-b7")
    model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # mode evaluasi
    return model