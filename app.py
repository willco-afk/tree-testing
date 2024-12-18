import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model_path = 'your_trained_model.keras'
model = load_model(model_path)

# Function to make predictions
def predict(image):
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)  # Make prediction
    return prediction

# Streamlit app layout
st.title("Image Classifier")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Show uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    prediction = predict(image)
    
    # Display prediction
    st.write(f"Prediction: {prediction}")