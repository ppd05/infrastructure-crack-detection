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


# ================= SESSION STATE =================
if "model" not in st.session_state:
    st.session_state.model = None

if "localizer" not in st.session_state:
    st.session_state.localizer = None


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Infrastructure Crack Detection",
    page_icon="🔍",
    layout="wide"
)


# ================= MODEL LOADER =================
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


# ================= MAIN APP =================
def main():
    st.title("🔍 Infrastructure Crack Detection")
    st.write("Upload an image to detect cracks with advanced AI analysis")

    analyzer = CrackSeverityAnalyzer()
    report_gen = ReportGenerator()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    # -------- Lazy model loading --------
    if uploaded_file and st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_crack_model()
            st.session_state.localizer = CrackLocalizer(
                st.session_state.model
            )

    if not uploaded_file:
        return

    # -------- Save temp image --------
    temp_img_path = "temp_image.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = Image.open(uploaded_file)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_column_width=True)

    # -------- Preprocess --------
    img_array = image.load_img(temp_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # -------- Predict --------
    with st.spinner("🔄 Analyzing image..."):
        prediction = st.session_state.model.predict(img_array, verbose=0)

    has_crack = prediction[0][0] > 0.5
    confidence = prediction[0][0] if has_crack else (1 - prediction[0][0])

    st.write("---")

    # ================= CRACK DETECTED =================
    if has_crack:
        st.error(f"🔴 **CRACK DETECTED** — Confidence: {confidence*100:.2f}%")

        heatmap_img, _ = st.session_state.localizer.generate_heatmap(temp_img_path)
        bbox_img, bboxes = st.session_state.localizer.detect_crack_bbox(temp_img_path)

        with col2:
            st.subheader("🔥 Crack Heatmap")
            st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col3:
            st.subheader("📦 Crack Location")
            st.image(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.write("---")
        st.subheader("📊 Crack Severity Analysis")

        severity = analyzer.assess_severity(heatmap_img, bboxes)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Severity Level", f"{severity['color_code']} {severity['severity_level']}")
        m2.metric("Severity Score", f"{severity['severity_score']:.1f}/100")
        m3.metric("Crack Area", f"{severity['area_percentage']:.2f}%")
        m4.metric("Avg Width", f"{severity['estimated_width']:.1f}px")

        # -------- PDF GENERATION --------
        st.write("---")
        st.subheader("📄 Generate PDF Report")

        if st.button("📥 Generate PDF Report"):
            with st.spinner("Creating PDF report..."):
                temp_original = "temp_original.jpg"
                temp_heatmap = "temp_heatmap.jpg"
                temp_bbox = "temp_bbox.jpg"

                cv2.imwrite(
                    temp_original,
                    cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(temp_heatmap, heatmap_img)
                cv2.imwrite(temp_bbox, bbox_img)

                pdf_path = report_gen.generate_report(
                    filename=uploaded_file.name,
                    has_crack=True,
                    confidence=confidence,
                    severity_result=severity,
                    original_img_path=temp_original,
                    heatmap_img_path=temp_heatmap,
                    bbox_img_path=temp_bbox
                )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    f,
                    file_name=f"crack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

            for f in [temp_original, temp_heatmap, temp_bbox]:
                if os.path.exists(f):
                    os.remove(f)

    # ================= NO CRACK =================
    else:
        st.success(f"🟢 **NO CRACK DETECTED** — Confidence: {confidence*100:.2f}%")

        st.write("---")
        if st.button("📥 Generate PDF Report (No Crack)"):
            with st.spinner("Creating PDF report..."):
                temp_original = "temp_original.jpg"
                cv2.imwrite(
                    temp_original,
                    cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
                )

                pdf_path = report_gen.generate_report(
                    filename=uploaded_file.name,
                    has_crack=False,
                    confidence=confidence,
                    original_img_path=temp_original
                )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    f,
                    file_name=f"no_crack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

            os.remove(temp_original)

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)


# ================= RUN =================
if __name__ == "__main__":
    main()
