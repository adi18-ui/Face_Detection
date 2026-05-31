"""
Face Detection System using OpenCV and Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Face Detection System", layout="centered")

st.title("🔍 Face Detection System")
st.markdown("Upload an image or use your webcam to detect faces using OpenCV's Haar Cascade classifier.")

# Load the pre-trained face detection model
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_cascade()

# Sidebar settings - Defaults updated for better crowd detection
st.sidebar.header("⚙️ Detection Settings")
scale_factor = st.sidebar.slider("Scale Factor", 1.01, 2.0, 1.1, 0.01) 
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 15, 4) 
min_size = st.sidebar.slider("Min Face Size (px)", 10, 200, 20) 
box_color = st.sidebar.color_picker("Box Color", "#00FF00")
box_thickness = st.sidebar.slider("Box Thickness", 1, 5, 2)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    return faces

def draw_faces(image, faces):
    output = image.copy()
    color = hex_to_bgr(box_color)
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, box_thickness)
        cv2.putText(output, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output

# --- Tabs ---
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Camera"])

with tab1:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        faces = detect_faces(image)
        result = draw_faces(image, faces)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            # Updated to use_container_width to fix the yellow warning
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader(f"Detected: {len(faces)} face(s)")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download result (Added unique key="dl_tab1")
        _, buf = cv2.imencode(".png", result)
        st.download_button(
            "⬇️ Download Result", 
            data=BytesIO(buf.tobytes()), 
            file_name="detected_faces.png", 
            mime="image/png",
            key="dl_tab1" 
        )

with tab2:
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        file_bytes = np.frombuffer(camera_input.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        faces = detect_faces(image)
        result = draw_faces(image, faces)

        st.subheader(f"Detected: {len(faces)} face(s)")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download result (Added unique key="dl_tab2")
        _, buf = cv2.imencode(".png", result)
        st.download_button(
            "⬇️ Download Result", 
            data=BytesIO(buf.tobytes()), 
            file_name="detected_camera_faces.png", 
            mime="image/png",
            key="dl_tab2"
        )

st.markdown("---")
st.caption("Built with OpenCV (Haar Cascade) & Streamlit")

