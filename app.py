from PIL import Image
import numpy as np
import streamlit as st
import torch
from model import DigitCNN

st.title("✍️ Handwritten Digit Recognition")

model = DigitCNN()
model.load_state_dict(torch.load("digit_model.pth", map_location="cpu"))
model.eval()

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    image = np.array(image)

    image = 255 - image
    image = image / 255.0

    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output).item()

    st.image(image, caption="Processed Image", width=150)
    st.success(f"Predicted Digit: **{prediction}**")
