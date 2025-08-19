# Cardiac Abnormality Detection from CT & MRI Scans using CNN & LSTM

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen)](#results)

## Overview
This project implements a **hybrid CNN-LSTM model** for automatic detection of cardiovascular abnormalities from CT and MRI scans.  
- **CNN:** Extracts spatial features from images.  
- **LSTM:** Captures temporal patterns across sequences of scans.  
- **Deployment:** Google Colab for training, Streamlit for real-time inference.  

---

## Dataset
- **CAD Cardiac Dataset** (Kaggle) – 65,000 images, 2 classes (`sick`, `normal`)  
- **Preprocessing:** Resize (128x128), grayscale conversion, normalization  

---

## Features
- High classification accuracy: **95%**  
- Evaluation metrics: Precision 0.943, Recall 0.952, F1-Score 0.947, AUC 0.98  
- Web-based interface via **Streamlit** for live testing  

---

## Installation
1. Clone this repository:
bash
git clone https://github.com/harshavardhan5423/Cardiac-Abnormality-Detection-from-CT-MRI-scans-using-CNN-LSTM.git
cd Cardiac-Abnormality-Detection-from-CT-MRI-scans-using-CNN-LSTM
Create a virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies:
pip install -r requirements.txt
Usage
Launch the Streamlit app:
streamlit run app.py
Upload CT/MRI scan images via the web interface.
Get real-time predictions for cardiovascular abnormalities.
Model Architecture
CNN Layers: Convolution + ReLU + MaxPooling for spatial feature extraction
LSTM Layer: 64 units for temporal pattern recognition
Output Layer: Dense with sigmoid activation for binary classification
Results
Accuracy (Test): 95%
Precision: 0.943
Recall: 0.952
F1-Score: 0.947
AUC Score: 0.98
Comparison:
Model	Accuracy
CNN-only	90%
CNN-LSTM	95%
Authors
Dokula Harsha Vardhan – dokulaharshavardhan@gmail.com
Gandham Sandeep – sandeepgandham03@gmail.com
Aringi Satish – satish22507@gmail.com
Kuna Raghu Ram – raghuramkuna2003@gmail.com
Mrs. N. R. S. L. Prasanthi – narayanam.prasanthi@gmail.com
Pitchuka Jyothsna Naga Mahalakshmi – nagamaha2003@gmail.com
