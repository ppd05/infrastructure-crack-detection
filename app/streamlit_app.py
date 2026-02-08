import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import requests
from PIL import Image

st.title("Infrastructure Crack Detection")
st.write("Upload an image of concrete or metal structure, and the app will detect cracks using the trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Detect Crack"):
        with st.spinner("Detecting crack..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                response.raise_for_status()
                result = response.json()
                pred_label = result.get("predicted_label", "Unknown")
                pred_prob = result.get("probability", 0.0)

                if pred_label == "crack":
                    color = "red"
                else:
                    color = "green"
                st.markdown(
                    f"<span style='color: {color}; font-weight: bold;'>"
                    f"Your uploaded image has: {pred_label.upper()} with probability {pred_prob:.4f}"
                    f"</span>", 
                    unsafe_allow_html=True,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload an image first.")