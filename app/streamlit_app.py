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
import os
from datetime import datetime
from src.crack_localization import CrackLocalizer
from src.severity_analyzer import CrackSeverityAnalyzer
from src.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Infrastructure Crack Detection",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_crack_model():
    model = load_model('models/resnet101_crack_detector.h5')
    return model

def main():
    # Header
    st.title("🔍 Infrastructure Crack Detection")
    st.write("Upload an image to detect cracks with advanced AI analysis")
    
    # Load model and initializers
    model = load_crack_model()
    localizer = CrackLocalizer(model)
    analyzer = CrackSeverityAnalyzer()
    report_gen = ReportGenerator()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of concrete or metal infrastructure"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_img_path = "temp_image.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and display original image
        img = Image.open(uploaded_file)
        
        # Create columns for image display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(img, use_column_width=True)
        
        # Preprocess image for prediction
        img_array = image.load_img(temp_img_path, target_size=(224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        with st.spinner("🔄 Analyzing image..."):
            prediction = model.predict(img_array, verbose=0)
        
        # Determine if crack is detected
        has_crack = prediction[0][0] > 0.5
        confidence = prediction[0][0] if has_crack else (1 - prediction[0][0])
        
        # Display results
        st.write("---")
        
        if has_crack:
            # CRACK DETECTED
            st.error(f"🔴 **CRACK DETECTED** - Confidence: {confidence*100:.2f}%")
            
            # Generate heatmap
            with st.spinner("🗺️ Generating crack heatmap..."):
                heatmap_img, _ = localizer.generate_heatmap(temp_img_path)
            
            with col2:
                st.subheader("🔥 Crack Heatmap")
                st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Generate bounding box
            with st.spinner("📍 Detecting crack location..."):
                bbox_img, bboxes = localizer.detect_crack_bbox(temp_img_path)
            
            with col3:
                st.subheader("📦 Crack Location")
                st.image(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Display bounding box coordinates
            if bboxes:
                st.write("**🎯 Detected Crack Regions:**")
                for i, (x, y, w, h) in enumerate(bboxes):
                    st.write(f"  • Region {i+1}: Position(x={x}, y={y}), Size(width={w}, height={h})")
            
            # Severity Analysis
            st.write("---")
            st.subheader("📊 Crack Severity Analysis")
            
            with st.spinner("⚖️ Analyzing severity..."):
                severity_result = analyzer.assess_severity(heatmap_img, bboxes)
            
            # Display severity metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Severity Level", 
                    f"{severity_result['color_code']} {severity_result['severity_level']}"
                )
            
            with metric_col2:
                st.metric("Severity Score", f"{severity_result['severity_score']:.1f}/100")
            
            with metric_col3:
                st.metric("Crack Area", f"{severity_result['area_percentage']:.2f}%")
            
            with metric_col4:
                st.metric("Avg Width", f"{severity_result['estimated_width']:.1f}px")
            
            # Recommendation box
            if severity_result['severity_level'] == 'HIGH':
                st.error(f"⚠️ **{severity_result['recommendation']}**")
            elif severity_result['severity_level'] == 'MEDIUM':
                st.warning(f"⚡ **{severity_result['recommendation']}**")
            else:
                st.info(f"ℹ️ **{severity_result['recommendation']}**")
            
            # Detailed measurements in expander
            with st.expander("📏 View Detailed Measurements"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write("**Physical Measurements:**")
                    st.write(f"- Estimated Length: {severity_result['estimated_length']:.1f} pixels")
                    st.write(f"- Estimated Width: {severity_result['estimated_width']:.1f} pixels")
                    st.write(f"- Affected Area: {severity_result['area_percentage']:.2f}%")
                
                with detail_col2:
                    st.write("**Severity Breakdown:**")
                    st.write(f"- Base Score: {severity_result['severity_score']:.1f}/100")
                    st.write(f"- Classification: {severity_result['severity_level']}")
                    st.write(f"- Risk Level: {severity_result['color_code']}")
            
            # PDF Report Generation
            st.write("---")
            st.subheader("📄 Generate Detailed Report")
            
            col_report1, col_report2 = st.columns([2, 1])
            
            with col_report1:
                st.write("Generate a comprehensive PDF report with all analysis results, images, and recommendations.")
            
            with col_report2:
                if st.button("📥 Generate PDF Report", use_container_width=True):
                    with st.spinner("📝 Creating PDF report..."):
                        # Save temporary images
                        temp_original = "temp_original.jpg"
                        temp_heatmap = "temp_heatmap.jpg"
                        temp_bbox = "temp_bbox.jpg"
                        
                        cv2.imwrite(temp_original, cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(temp_heatmap, heatmap_img)
                        cv2.imwrite(temp_bbox, bbox_img)
                        
                        # Generate PDF
                        pdf_path = report_gen.generate_report(
                            filename=uploaded_file.name,
                            has_crack=True,
                            confidence=confidence,
                            severity_result=severity_result,
                            original_img_path=temp_original,
                            heatmap_img_path=temp_heatmap,
                            bbox_img_path=temp_bbox
                        )
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Download button
                    with open(pdf_path, 'rb') as f:
                        report_filename = f"crack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=f,
                            file_name=report_filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # Cleanup temporary files
                    for temp_file in [temp_original, temp_heatmap, temp_bbox]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
        
        else:
            # NO CRACK DETECTED
            st.success(f"🟢 **NO CRACK DETECTED** - Confidence: {confidence*100:.2f}%")
            
            with col2:
                st.subheader("✅ Analysis Complete")
                st.info("The infrastructure appears to be in good condition with no visible cracks detected.")
            
            with col3:
                st.subheader("📋 Recommendation")
                st.write("**Status:** Healthy")
                st.write("**Action:** Continue regular monitoring")
                st.write("**Next Check:** As per maintenance schedule")
            
            # Option to generate PDF for no-crack cases too
            st.write("---")
            if st.button("📥 Generate PDF Report (No Crack)", use_container_width=False):
                with st.spinner("📝 Creating PDF report..."):
                    temp_original = "temp_original.jpg"
                    cv2.imwrite(temp_original, cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR))
                    
                    pdf_path = report_gen.generate_report(
                        filename=uploaded_file.name,
                        has_crack=False,
                        confidence=confidence,
                        original_img_path=temp_original
                    )
                
                st.success("✅ Report generated!")
                
                with open(pdf_path, 'rb') as f:
                    report_filename = f"no_crack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=f,
                        file_name=report_filename,
                        mime="application/pdf"
                    )
                
                if os.path.exists(temp_original):
                    os.remove(temp_original)
        
        # Cleanup main temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        **Infrastructure Crack Detection System**
        
        This AI-powered tool uses deep learning (ResNet101) to:
        - ✅ Detect cracks in infrastructure
        - 📍 Localize crack positions
        - ⚖️ Assess severity levels
        - 📄 Generate detailed reports
        
        **Accuracy:** 95%+
        **Speed:** < 2 seconds
        """)
        
        st.header("📚 How to Use")
        st.write("""
        1. Upload an image (JPG/PNG)
        2. Wait for AI analysis
        3. Review results and severity
        4. Download PDF report
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for preliminary assessment only. 
        Always consult professional structural engineers 
        for critical infrastructure decisions.
        """)
        
        st.header("🔧 Tech Stack")
        st.code("""
        • TensorFlow/Keras
        • ResNet101
        • OpenCV
        • Streamlit
        • ReportLab
        """)

if __name__ == "__main__":
    main()