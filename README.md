
cd fast-alpr

2. Install Dependencies

For CPU only:

pip install fast-alpr[onnx]


For NVIDIA GPU:

pip install fast-alpr[onnx-gpu]

3. Test the Library
from fast_alpr import ALPR

# Load ALPR with default models
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Run prediction on a sample image
print(alpr.predict("assets/test_image.png"))


Expected output (example):

[
  {
    "plate": "ABC123",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.97
  }
]

4. Build a Streamlit App

Create a file app.py:

import streamlit as st
from fast_alpr import ALPR
import cv2
import tempfile
import os

# Load ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

st.title("🚘 FastALPR - License Plate Recognition")

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        img_path = tmp_file.name
    
    # Run ALPR
    results = alpr.predict(img_path)

    # Show image + results
    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    st.json(results)

    # Cleanup
    os.remove(img_path)

5. Run the App
streamlit run app.py
