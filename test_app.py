import streamlit as st
import cv2
import numpy as np
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

# Minimal FER test
fer_status = "FER disabled for now"  # Placeholder
try:
   from fer import fer
   detector = fer.FER()  # Basic instantiation, no MTCNN yet
   fer_status = "Success! FER loaded and detector created"
except Exception as e:
   fer_status = f"FER failed: {str(e)}"

st.set_page_config(page_title="Patent Project Test", layout="wide")

st.title("Setup Validation – Patent Project 🚀")
st.write("Python & packages check:")

st.success(f"OpenCV version: {cv2.__version__}")
st.info(fer_status)
st.success(f"NumPy version: {np.__version__}")
st.success(f"Torch version: {torch.__version__}")

st.success("If you see this, the page is not blank!")
st.balloons()