from keras import models
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import os


model_path = "C:/Users/Acer/Desktop/test/V2/saved.h5"
test_images_path = "C:/Users/Acer/Desktop/test/V2/imgs"

mod = models.load_model(model_path)

img_data = st.file_uploader(label = "Input Image", type = ['jpg'])

if img_data is not None:

    up_img = Image.open(img_data)
    st.image(up_img)

    img_data.name = "test.jpeg"
    
    with open(os.path.join(test_images_path, img_data.name), "wb") as f:
        f.write((img_data).getbuffer())
    imgs = []
    for image in os.listdir(test_images_path):
        img_path = os.path.join(test_images_path, image)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (100, 100))
        imgs.append(img_arr / 255)
    
    imgs = np.array(imgs)
    
    pred = mod.predict(imgs)
    
    for dis in pred:
        if(dis[0] > dis[1]):
            st.write("CAT")
        else:
            st.write("DOG")
