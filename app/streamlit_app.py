import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from datetime import datetime

from src.crack_localization import CrackLocalizer
from src.severity_analyzer import CrackSeverityAnalyzer
from src.report_generator import ReportGenerator


# ---------------- SESSION STATE INIT ----------------
if "model" not in st.session_state:
    st.session_state.model = None

if "localizer" not in st.session_state:
    st.session_state.localizer = None


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Infrastructure Crack Detection",
    page_icon="🔍",
    layout="wide"
)


# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_crack_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(
        base_dir,
        "..",
        "models",
        "resnet101_crack_detector.h5"
    )
    return load_model(model_path)


# ---------------- MAIN APP ----------------
def main():
    st.title("🔍 Infrastructure Crack Detection")
    st.write("Upload an image to detect cracks with advanced AI analysis")

    analyzer = CrackSeverityAnalyzer()
    report_gen = ReportGenerator()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of concrete or metal infrastructure"
    )

    # -------- LOAD MODEL LAZILY --------
    if uploaded_file is not None and st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_crack_model()
            st.session_state.localizer = CrackLocalizer(
                st.session_state.model
            )

    if uploaded_file is None:
        return

    # -------- SAVE TEMP IMAGE --------
    temp_img_path = "temp_image.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = Image.open(uploaded_file)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_column_width=True)

    # -------- PREPROCESS --------
    img_array = image.load_img(temp_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # -------- PREDICTION --------
    with st.spinner("🔄 Analyzing image..."):
        prediction = st.session_state.model.predict(img_array, verbose=0)

    has_crack = prediction[0][0] > 0.5
    confidence = prediction[0][0] if has_crack else (1 - prediction[0][0])

    st.write("---")

    # ================= CRACK DETECTED =================
    if has_crack:
        st.error(f"🔴 **CRACK DETECTED** - Confidence: {confidence*100:.2f}%")

        with st.spinner("🗺️ Generating crack heatmap..."):
            heatmap_img, _ = st.session_state.localizer.generate_heatmap(temp_img_path)

        with col2:
            st.subheader("🔥 Crack Heatmap")
            st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        with st.spinner("📍 Detecting crack location..."):
            bbox_img, bboxes = st.session_state.localizer.detect_crack_bbox(temp_img_path)

        with col3:
            st.subheader("📦 Crack Location")
            st.image(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.write("---")
        st.subheader("📊 Crack Severity Analysis")

        severity_result = analyzer.assess_severity(heatmap_img, bboxes)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Severity Level", f"{severity_result['color_code']} {severity_result['severity_level']}")
        m2.metric("Severity Score", f"{severity_result['severity_score']:.1f}/100")
        m3.metric("Crack Area", f"{severity_result['area_percentage']:.2f}%")
        m4.metric("Avg Width", f"{severity_result['estimated_width']:.1f}px")

    # ================= NO CRACK =================
    else:
        st.success(f"🟢 **NO CRACK DETECTED** - Confidence: {confidence*100:.2f}%")

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
