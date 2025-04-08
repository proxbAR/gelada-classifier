import streamlit as st

st.title('Gelada Classifier')

st.info("This is a Gelada classifier Machine Learning application!")

image_uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
