import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Constants
MODEL_PATH = "mlp_model.tflite"
IMG_SIZE = (64, 64)  # adjust to your model input size
CLASS_NAMES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
               "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
               "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
               "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
               # Add all class names here...
              ]

# Load TFLite model
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(MODEL_PATH)

def preprocess_image(image: Image.Image):
    # Resize and normalize the image to [0,1]
    img = image.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32) / 255.0
    # Flatten for MLP (assumes MLP input is 1D)
    img = img.flatten()
    # Add batch dimension
    return np.expand_dims(img, axis=0)

def predict(image: Image.Image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]
    predicted_class = CLASS_NAMES[predicted_index]
    
    return predicted_class, confidence

# Streamlit app UI
st.title("Plant Disease Detection - MLP with TFLite Model")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            label, conf = predict(image)
        st.success(f"Prediction: **{label}** (Confidence: {conf:.2f})")
else:
    st.info("Please upload an image file to start prediction.")
