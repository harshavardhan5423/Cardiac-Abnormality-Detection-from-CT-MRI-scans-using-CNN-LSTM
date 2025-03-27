import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Google Drive File ID
file_id = "1Wcizk9nXzhZnvZXlfhCIv1AeK3H2Uy4A"  # Replace with your file ID
url = f"https://drive.google.com/uc?id={file_id}"
output = "my_model.keras"  # Local filename for the model

# Download the model file from Google Drive (only needed if model is not present locally)
gdown.download(url, output, quiet=False)

# Load the model
model = tf.keras.models.load_model(output)

# Define class labels (adjust as needed)
class_names = ["No Disease", "Disease"]

# Function to preprocess the image (adjusted to fit input shape)
def preprocess_image(image):
    # Resize image to match model input size
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image) / 255.0  # Normalize image values to between 0 and 1
    
    # If model expects 10 time steps, we simulate it by repeating the same image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (grayscale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (shape: 1, 128, 128, 1)
    
    # Now repeat the image 10 times to simulate time steps
    image = np.repeat(image, 10, axis=1)  # Repeat along the time axis
    
    image = np.expand_dims(image, axis=0)  # Add final batch dimension, making shape (1, 10, 128, 128, 1)
    return image

# Streamlit UI
st.title("Disease Detection ML Model")
st.write("Upload an image to check for disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Display result
    result = class_names[int(prediction[0] > 0.5)]
    st.write(f"### Prediction: {result}")
