import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Define class names (adjust these as per your dataset)
class_names = ['No Disease', 'Disease']

# Preprocess the input image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize the image to match model input
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Set up the Streamlit app UI
st.title("Cardiovascular Disease Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(img)
    
    # Make the prediction
    prediction = model.predict(processed_image)
    
    # Display result
    result = class_names[int(prediction[0] > 0.5)]  # If prediction > 0.5, it's Disease; else No Disease
    st.write(f"Prediction: {result}")
    
    # Optionally, display the confidence score
    confidence = prediction[0] * 100
    st.write(f"Confidence: {confidence:.2f}%")

