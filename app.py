import streamlit as st
import torch
import cv2
import numpy as np
from model import DigitCNN  # your model class

st.title("✍️ Handwritten Digit Recognition")

model = DigitCNN()
model.load_state_dict(torch.load("digit_model.pth", map_location="cpu"))
model.eval()

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (28, 28))
    image = 255 - image
    image = image / 255.0

    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output).item()

    st.image(image, caption="Processed Image", width=150)
    st.success(f"Predicted Digit: **{prediction}**")
