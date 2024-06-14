import streamlit as st
import torch
import torch.nn as nn
from model import Generator
import config
from PIL import Image
import os
import numpy as np
from torchvision.utils import save_image
from test import load_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

test_transform = A.Compose(
    [
      A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
      ToTensorV2(),
    ]
)

def process_image(img):
  np_image = np.asarray(Image.open(img).convert("RGB"))
  processed = test_transform(image = np_image)["image"].unsqueeze(0).to(config.DEVICE)
  return processed

def upscale(processed_img):
  with torch.no_grad():
    high_res = st.session_state['model'](processed_img)
  high_res = np.asarray(high_res*0.5+0.5).squeeze()
  high_res = np.moveaxis(high_res, 0, 2)
  return high_res

st.session_state['model'] = load_model()
# st.text(st.session_state['model'])
img = st.file_uploader("upload image", type=['jpg', 'png', 'jpeg'])
left, right = st.columns(2)
if img:
  processed = process_image(img)
  high_res = upscale(processed)
  st.image(img, 'Old image', use_column_width=True)
  st.image(high_res, 'Upscaled', use_column_width=True)