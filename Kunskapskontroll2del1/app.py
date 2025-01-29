import streamlit as st
import numpy as np
import cv2
from joblib import load
from PIL import Image, ImageOps
import tempfile
from streamlit_drawable_canvas import st_canvas

# Load saved model
model = load('mnist_model.joblib')

# Function for processing picture
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = cv2.bitwise_not(resized)
    normalized = inverted / 255.0
    flattened = normalized.reshape(1, -1)
    return flattened

# Streamlit UI
st.title("MNIST Sifferigenkänning")

# Load picture
st.header("1. Ladda upp en bild")
uploaded_file = st.file_uploader("Välj en bild", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uppladdad bild", use_column_width=True)
    img_array = np.array(img)
    processed_img = preprocess_image(img_array)
    prediction = model.predict(processed_img)
    st.write(f"Predikterad siffra: **{prediction[0]}**")

# Draw number
st.header("2. Rita en siffra")
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Check if the user has drawn something
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="Ritad bild", use_column_width=False)
    processed_img = preprocess_image(canvas_result.image_data[:, :, 0])
    prediction = model.predict(processed_img)
    st.write(f"Predikterad siffra: **{prediction[0]}**")

# Camera
st.header("3. Ta en bild med kameran")
camera_image = st.camera_input("Ta en bild med din kamera")
if camera_image is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(camera_image.read())
    img = Image.open(temp_file.name)
    st.image(img, caption="Kamerabild", use_column_width=True)
    img_array = np.array(img)
    processed_img = preprocess_image(img_array)
    prediction = model.predict(processed_img)
    st.write(f"Predikterad siffra: **{prediction[0]}**")