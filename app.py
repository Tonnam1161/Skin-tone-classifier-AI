import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="AI detective", layout="centered")
st.header("AI Detective : Colour Skin Person")
st.caption("Upload image...")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

uploaded_file = st.file_uploader("Upload Image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Image")
        st.image(image)

    if st.button("Detect Objects"):
        with st.spinner("Ai Detecting .."):
            results = model(image)
            res_plotted = results[0].plot()
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.write("Ai result")
                st.image(res_plotted)
            
            top1_index = results[0].probs.top1
            class_name = results[0].names[top1_index]

            st.success(f"Results: {class_name}")