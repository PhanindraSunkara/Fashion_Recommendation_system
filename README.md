# Fashion_Recommendation_system
Fashion Recommendation System

#Introduction

The Fashion Recommendation System is an AI-based application designed to suggest visually similar fashion products based on an uploaded image. It utilizes Deep Learning (ResNet50) for feature extraction and k-Nearest Neighbors (k-NN) for similarity search, ensuring accurate and efficient recommendations.

#Features

Users can upload an image to find similar fashion products.
The system employs ResNet50, a pre-trained Convolutional Neural Network (CNN), for feature extraction.
It uses k-NN with Euclidean distance to find top-5 similar products.
Precomputed feature vectors allow for fast and real-time recommendations.
The system includes an interactive web application built with Streamlit for a seamless user experience.


#Tech Stack

The project is built using:
Python for backend processing
TensorFlow/Keras for deep learning (ResNet50 feature extraction)
scikit-learn for k-NN similarity search
Streamlit for the web application
OpenCV and NumPy for image processing
Pandas for dataset handling


#Dataset

The model is trained on the Fashion Product Images Dataset available on Kaggle. This dataset contains diverse categories of fashion items, making it suitable for recommendation systems.
📌 You can download the dataset from: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

#Installation Guide

Follow these steps to set up and run the Fashion Recommendation System:

Step 1: Clone the Repository
Open your terminal or command prompt and run:
git clone https://github.com/PhanindraSunkara/Fashion_Recommendation_System.git  
cd Fashion-Recommendation-System

Step 2: Install Required Dependencies
Ensure you have Python 3.7+ installed, then run:
pip install -r requirements.txt

Step 3: Download and Extract Dataset
Download the dataset from Kaggle and place it inside the project folder.

Step 4: Run Feature Extraction and Model Training
This step runs only once to extract features and train the k-NN model:
python main.py

Step 5: Start the Web Application
Run the following command to launch the Streamlit web app:
streamlit run app.py

How It Works

1. Users upload an image of a fashion product through the web app.
2. The system extracts deep features from the image using ResNet50.
3. A k-NN similarity search retrieves the top-5 matching products based on Euclidean distance.
4. The recommended products are displayed in the web application.



Results

The system provides accurate fashion recommendations based on deep image features. Below is an example of the output:


📌 The uploaded shoe image is processed, and the most visually similar shoes are displayed in the web app.

Future Improvements

To enhance the system, the following improvements can be implemented:
🚀 Use Approximate Nearest Neighbors (ANN) for faster similarity search.
🚀 Integrate text-based attributes (e.g., brand, color, category) to enhance recommendations.
🚀 Deploy on Cloud Platforms (AWS, Google Cloud) for better scalability and performance.
🚀 Expand to more fashion categories like accessories, eyewear, and clothing.

Contributing

We welcome contributions! If you’d like to improve this project:

1. Fork the repository
2. Create a new branch
3. Make your changes and submit a pull request


Allowing free and open-source use.

👕👟 Happy Shopping with AI-powered Recommendations! 🎉


