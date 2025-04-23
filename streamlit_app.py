import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("https://github.com/kanagathirsha/CottonLeafDiseasePrediction/releases/download/v1.0.0/best_densenet_model.h5")

# Class names (same order as your training)
class_labels = ['Aphids', 'Army Worm', 'Bacterial Blight', 'Healthy Leaf', 'Powdery Mildew', 'Target Spot']

# App UI
st.title("Cotton Leaf Disease Prediction")
st.write("Upload a cotton leaf image to classify the disease.")

# Upload image
uploaded_file = st.file_uploader("CHOOSE IMAGE....", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown(f" Prediction:{predicted_label}")
    st.markdown(f" Confidence:{confidence:.2f}%")
