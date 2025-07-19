import os
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras import models

model = models.load_model("cxr_normal_tb_vgg16_model.keras")

model_xray_or_not = models.load_model("cxr_or_not_vgg16_model.keras")

def process_image(path_to_image):
    img = Image.open(path_to_image)
    img_one = img.convert("RGB")
    img_two = img_one.resize((156,156))
    img_data = np.asarray(img_two)
    img_data_normalized = img_data/255
    return img_data_normalized

def is_xray(model, path_to_image):
    img_data_normalized = process_image(path_to_image)
    xray = model.predict(np.asarray([img_data_normalized]))
    print(xray)
    if xray > 0.5:
        return True
    else:
        return False
    
def predict_image(model, path_to_image):
    img_data_normalized = process_image(path_to_image)
    is_xray = is_xray(model_xray_or_not, path_to_image)
    if is_xray:
        img_data_normalized = process_image(path_to_image)
        prob_of_tb = model.predict(np.asarray([img_data_normalized]))
        if prob_of_tb > 0.5:
            message = "TB is likely"
        elif prob_of_tb > 0.4:
            message = "TB cannot be ruled out"
        else:
            message = "TB is unlikely"
        return prob_of_tb, message
    else:
        prob_of_tb = np.array([[0.0]])
        message = "This is not an X-ray"
        return prob_of_tb, message
    
st.title("Normal vs TB CXR App")
st.citations("This app predicts if a picture of a CXR is likely to be TB or not")
uploaded_files = st.file_uploader("CXR Picture", accept_multiple_files=True)