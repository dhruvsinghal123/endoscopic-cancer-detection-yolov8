import streamlit as st
import os
import shutil
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model_path = "best.pt"

model = YOLO(model_path)

# Streamlit Page Config
st.set_page_config(page_title="Throat Cancer Detector", layout="wide")

# Title
st.title("🔬 Throat Cancer Detection from Endoscopy Images")
st.markdown("Upload your endoscopy images below to check for **benign** or **malignant** conditions.")

# Upload UI
uploaded_files = st.file_uploader("📤 Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    test_dir = "test_uploads"
    results_dir = "detection_results"

    # Cleanup or create folders
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # Save uploaded files
    for uploaded_file in uploaded_files:
        with open(os.path.join(test_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"{len(uploaded_files)} image(s) uploaded successfully!")

    # Run YOLOv8 on images
    with st.spinner("Detecting..."):
        results = model.predict(source=test_dir, save=True, project=results_dir, name="run", conf=0.25)

    # Display annotated images
    output_path = os.path.join(results_dir, "run")
    st.subheader("🩺 Detection Results")
    image_cols = st.columns(3)

    for idx, file in enumerate(os.listdir(output_path)):
        if file.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(output_path, file))
            with image_cols[idx % 3]:
                st.image(img, caption=file, use_column_width=True)
