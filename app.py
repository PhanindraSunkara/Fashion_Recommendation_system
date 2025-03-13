import streamlit as st
import pickle
import numpy as np
import cv2
import tensorflow as tf
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load k-NN model
knn_model = pickle.load(open('knn_model.pkl', 'rb'))


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)


def recommend_images(img_path):
    normalized_result = extract_features(img_path)
    distances, indices = knn_model.kneighbors([normalized_result])
    return [filenames[i] for i in indices[0][1:6]]


# Streamlit UI
st.title("Image Recommendation System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    recommended_images = recommend_images(file_path)

    st.subheader("Recommended Images")
    cols = st.columns(5)
    for col, img_path in zip(cols, recommended_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col.image(img, use_column_width=True)
