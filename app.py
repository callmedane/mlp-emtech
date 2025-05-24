import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.title("🌿 Plant Disease Classifier (MLP Model)")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = img_to_array(image) / 255.0
    img_array = img_array.reshape(1, -1)

    model = load_model("mlp_model.h5")
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    # Replace with your label encoder classes
    classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', ...]  # Complete this
    st.success(f"Prediction: {classes[class_idx]}")
