import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Grading App", layout="wide")
st.title("📘 Automatic Grading App")

st.write("Upload an answer sheet image to scan:")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=400)

    st.success("Image uploaded successfully! (This is a placeholder — grading logic will be added later.)")
else:
    st.info("Please upload an image to begin.")
