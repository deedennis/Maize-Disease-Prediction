 🌽 Maize Disease Prediction System

 📌 Project Overview
This project is a deep learning-based application that detects maize leaf diseases from images.  
Users can upload an image of a maize leaf and receive an instant prediction of the disease.

 🚀 Live Demo
👉 https://maize-disease-prediction-nhtewguuchm2chn4wiilac.streamlit.app/

 🎯 Objectives
- To develop a deep learning model for maize disease detection  
- To provide fast and accurate predictions  
- To assist farmers and agricultural experts in identifying crop diseases  

 ❗ Problem Statement
Maize farmers often face challenges in identifying plant diseases early and accurately.  
This leads to crop losses and reduced agricultural productivity.  
This project aims to solve this problem using artificial intelligence.

 💡 Proposed Solution
The system uses a trained deep learning model to classify maize leaf images into different disease categories.  
It provides a simple interface where users can upload images and get predictions instantly.

 📊 Dataset
- Image dataset of maize leaves  
- Categories include:
  - Healthy  
  - Common Rust  
  - Leaf Blight  
  - Gray Leaf Spot  
- Images were preprocessed through resizing and normalization  
- Dataset split into training and testing sets  

 🤖 Model Used
- Convolutional Neural Network (CNN)  
- Trained on labeled maize leaf images  
- Evaluated using accuracy and loss metrics  
- Best model selected based on performance  

 🖥️ Features
- Upload maize leaf image  
- Instant disease prediction  
- Simple and user-friendly interface  

 👥 Users
- Farmers  
- Agricultural extension officers  
- Researchers  

 ⚙️ Installation

```bash
git clone https://github.com/deedennis/Maize-Disease-Prediction.git
cd Maize-Disease-Prediction
pip install -r requirements.txt
streamlit run app.py
 📸 Screenshots

 🏠 Home Page
![Home](images/home.png)

 📤 Upload Image
![Upload](images/upload.png)

 🔍 Prediction Result
![Result](images/result.png)
