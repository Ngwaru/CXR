import os
import io
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras import models
from GradCAM import Grad_CAM_class
from typing import List, Tuple, Set, Any

model = models.load_model("cxr_normal_tb_vgg16_model.keras")

model_xray_or_not = models.load_model("cxr_or_not_vgg16_model.keras")

def process_image(path_to_image: str)->Any:
    """
    This function gets the image whose path is passed as a 
    string and returns the normalized numpy array.
    Parameters:
        path_to_image: 'pic.jpg'
    Returns:
        img_data_normalized: 
    """
    img = Image.open(path_to_image)
    img_one = img.convert("RGB")
    img_two = img_one.resize((200,200))
    img_data = np.asarray(img_two)
    img_data_normalized = img_data/255
    return img_data_normalized

def is_xray_func(model: Any, path_to_image: str) -> bool:
    """This function takes a tensorflow model and the path to the image 
    it returns true if the picture is an X ray
    Parameters:
        model: model
        path_to_image: "test_pic.png"
    Returns:
        bool: True
    """
    img_data_normalized = process_image(path_to_image)
    xray = model.predict(np.asarray([img_data_normalized]))
    if xray > 0.5:
        return True
    else:
        return False
    
def predict_image(model: Any, path_to_image: str)->tuple[List[List], str]:
    """This function gets a tensorflow model and path to an image and 
    returns a tuple with two compnents array and a string.
    Parameters:
        model: model
        path_to_image: "test_pic.png"
    Returns:
        (tuple[list[list]], "This is not an X-ray")
    """
    img_data_normalized = process_image(path_to_image)
    is_xray = is_xray_func(model_xray_or_not, path_to_image)
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



# def on_change(uploaded_files):
#     stream = io.BytesIO(uploaded_files.getbuffer())
#     prob_of_tb, message = predict_image(model, stream)
#     st.image(stream)
#     st.write(f"{message}")
    
st.title("Detect TB on CXR App")
st.caption("This app predicts if a picture of a CXR is likely to be TB or not")
uploaded_files = st.file_uploader("CXR Picture", accept_multiple_files=False, type=["jpg", "jpeg", "png"])
col_1, col_2, col_3 = st.columns(3)
with col_1:
    pass
with col_3:
    pass
with col_2:
    if st.button("Process Uploaded Picture"):
        stream = io.BytesIO(uploaded_files.getbuffer())
        prob_of_tb, message = predict_image(model, stream)
        prob_of_tb = int(prob_of_tb[0][0]*100)
        print(type(prob_of_tb))
        classes_names = {0:"Normal", 1:"Tuberculosis"}
        st.image(stream)
        with open("temp_img.png", "wb") as pic:
            pic.write(uploaded_files.getbuffer())
        img_opened = process_image(stream)
        grad_cam = Grad_CAM_class(model, "block5_conv4", "temp_img.png", classes_names, (200,200))
        grad_cam.get_img_array()
        grad_cam.make_gradcam_heatmap()
        superimposed_img = grad_cam.save_and_display_gracam()
        st.image(superimposed_img)
        message_html = f"""<h4 style='text_align: center; color: blue; font-family: Ariel, Helvetica, sans-serif;'>{message}</h4>"""
        prob_of_tb_html = f"""<h4 style='text-align: center; color: blue; fonat-family: Ariel, Helvetica, sans-serif;'>{prob_of_tb} % Probability of TB </h4>"""
        st.markdown(message_html, unsafe_allow_html=True)
        st.markdown(prob_of_tb_html, unsafe_allow_html=True)

