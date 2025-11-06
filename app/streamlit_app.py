import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model once and cache it
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("models/resnet101_crack_detector.h5")

model = load_model()

st.title("Crack Detection")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # Adjust to your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(img)
    prediction = model.predict(input_data)[0][0]  # Adjust if your model has different output shape

    if prediction > 0.5:  # Threshold for crack detection
        st.markdown('<p style="color:orange; font-weight:bold;">Crack Detected!</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-weight:bold;">No Crack Detected</p>', unsafe_allow_html=True)
