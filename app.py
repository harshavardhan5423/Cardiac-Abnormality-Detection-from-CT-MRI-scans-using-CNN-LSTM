import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Google Drive File ID
file_id = "1Wcizk9nXzhZnvZXlfhCIv1AeK3H2Uy4A"  # Replace with your file ID
url = f"https://drive.google.com/uc?id={file_id}"
output = "my_model.keras"  # Change to .h5 format

# Download the model file from Google Drive (only if not present locally)
gdown.download(url, output, quiet=False)

# ✅ Load the .h5 model
model = tf.keras.models.load_model(output)

# Define class labels
class_names = ["No Disease", "Disease"]

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale (if needed)
    img = img.resize((128, 128))  # Resize the image
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 128, 128, 1)
    return img_array

# Streamlit UI
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

    # Interpret results
    result = class_names[int(prediction[0][0] > 0.5)]  # If > 0.5, it's "Disease", else "No Disease"

    # Display results
    st.write(f"Prediction: **{result}**")
    
    # Show confidence score
    confidence = prediction[0][0] * 100
    st.write(f"Confidence: **{confidence:.2f}%**")
