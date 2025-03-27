import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Google Drive File ID
file_id = "1Wcizk9nXzhZnvZXlfhCIv1AeK3H2Uy4A"
url = f"https://drive.google.com/uc?id={file_id}"
output = "my_model.keras"  # Local filename for the model

# Download the model file from Google Drive
gdown.download(url, output, quiet=False)

# Load the model
model = tf.keras.models.load_model(output)

# Define class labels (adjust as needed)
class_names = ["No Disease", "Disease"]

# Function to preprocess the image
def preprocess_image(image):
    # Resize and convert to grayscale if needed, ensure it matches model input shape
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((128, 128))  # Adjust based on your model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
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
