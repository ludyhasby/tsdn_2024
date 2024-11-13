import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

def saliency_map(model, img, device, transform, pred):
  image = [transform(img.convert("RGB"))]
  image = torch.stack(image).to(device) # to GPU

  model.eval()
  model.to(device)
  image = image.to(device)

  # saliency
  image.requires_grad_()
  model.zero_grad() # clear any previous gradients
  outputs = model(image).logits # forward pass
  
  class_neuron = outputs[0, pred]
  class_neuron.backward() # backward
  gradients = image.grad[0].cpu().detach().numpy() # get gradients
  saliency = np.max(np.abs(gradients), axis=0) #gen map by taking max gradient across channel (max channel)
  fig, ax = plt.subplots()
  ax.imshow(saliency, cmap='hot')  # Use 'hot' colormap for better visualization
  ax.axis('off')  # Turn off axis
  st.pyplot(fig)
