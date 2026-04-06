import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("braille_model_final.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Title
st.title("🔤 Braille Character Recognition")
st.write("Upload a Braille image and get prediction")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Convert to grayscale
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Resize
        img = cv2.resize(img, (28,28))
        
        # Normalize
        img = img / 255.0
        
        # Reshape
        img = img.reshape(1,28,28,1)

        # Prediction
        probs = model.predict(img)
        pred = np.argmax(probs)
        label = le.inverse_transform([pred])[0]
        confidence = np.max(probs)

        # Output
        st.success(f"Predicted Character: {label}")
        st.info(f"Confidence: {confidence:.2f}")