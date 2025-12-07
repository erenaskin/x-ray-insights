import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import cv2
import sys
import os
import base64

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils_dicom import read_dicom

# API Config
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="X-Ray Insight", page_icon="🩻")

st.title("🩻 X-Ray Insight: Pneumonia & COVID-19 Detection")
st.markdown("""
This application detects **Normal**, **Pneumonia**, and **COVID-19** from chest X-ray images.
It supports **DICOM (.dcm)** and standard image formats (JPG, PNG).
""")

uploaded_file = st.file_uploader("Upload an X-Ray image...", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display Image
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    
    metadata = {}
    
    if filename.endswith('.dcm'):
        try:
            image, metadata = read_dicom(file_bytes)
            st.success("DICOM file successfully processed.")
            if metadata:
                with st.expander("Patient Information (Anonymous)"):
                    st.json(metadata)
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
            st.stop()
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
    # Option to invert image manually
    invert_image = st.checkbox("Görüntüyü Ters Çevir (Kemikler SİYAH ise işaretleyin)")
    
    if invert_image:
        # Invert image
        image_np = np.array(image)
        # Handle different channels
        image_np = 255 - image_np
        image = Image.fromarray(image_np)
        
        # Update file_bytes to send the inverted image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        
    with col1:
        st.image(image, caption='Uploaded Image', width="stretch")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing via API..."):
            try:
                # Send to API
                sent_filename = uploaded_file.name
                if invert_image:
                    sent_filename = "inverted_image.png"
                    
                files = {"file": (sent_filename, file_bytes, "image/png" if invert_image else uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")
                    probs = result.get("probabilities", {})
                    gradcam_b64 = result.get("gradcam_image")
                    
                    st.divider()
                    st.subheader("Sonuçlar")
                    
                    st.write(f"**Tahmin:** {prediction}")
                    st.write(f"**Güven Skoru:** %{confidence*100:.2f}")
                    
                    # Display detailed probabilities
                    if probs:
                        st.write("**Detaylı Olasılıklar:**")
                        st.progress(probs.get("NORMAL", 0), text=f"NORMAL: %{probs.get('NORMAL', 0)*100:.1f}")
                        st.progress(probs.get("PNEUMONIA", 0), text=f"PNEUMONIA: %{probs.get('PNEUMONIA', 0)*100:.1f}")
                        st.progress(probs.get("COVID-19", 0), text=f"COVID-19: %{probs.get('COVID-19', 0)*100:.1f}")
                    
                    if prediction == "NORMAL":
                        st.success(f"✅ TESPİT: {prediction}")
                    elif prediction == "PNEUMONIA":
                        st.warning(f"⚠️ TESPİT: {prediction}")
                    else:
                        st.error(f"🚨 TESPİT: {prediction}")
                        
                    # Display Grad-CAM if available
                    if gradcam_b64:
                        with col2:
                            gradcam_img = Image.open(io.BytesIO(base64.b64decode(gradcam_b64)))
                            st.image(gradcam_img, caption='Grad-CAM Analysis', width="stretch")
                    else:
                        gradcam_error = result.get("gradcam_error")
                        if gradcam_error:
                            st.warning(f"Grad-CAM visualization could not be generated. Error: {gradcam_error}")
                        else:
                            st.info("Grad-CAM visualization could not be generated.")
                    
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {e}. Ensure the API is running (uvicorn api.main:app).")

