import os
import pickle
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors

# Load Pre-trained ResNet50 Model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Function to Extract Features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result = model.predict(preprocessed_img, verbose=0)
    flattened = result.flatten()
    return flattened / norm(flattened)  # Normalize feature vector

# File Paths for Saving Data
features_file = "embedding.pkl"
images_file = "filenames.pkl"
knn_model_file = "knn_model.pkl"

# Extract & Save Features if Not Already Done
if not os.path.exists(features_file) or not os.path.exists(knn_model_file):
    print("Extracting features... This will run only once.")

    image_folder = r"C:\Users\saiku\PycharmProjects\pythonProject2\images"
    img_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    image_features = []
    for file in tqdm(img_files):
        features = extract_features(file, model)
        image_features.append(features)

    # Convert to NumPy array for efficiency
    image_features = np.array(image_features)

    # Save extracted features and image paths
    pickle.dump(image_features, open(features_file, "wb"))
    pickle.dump(img_files, open(images_file, "wb"))

    # Train and Save k-NN Model
    knn = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    knn.fit(image_features)
    pickle.dump(knn, open(knn_model_file, "wb"))

    print("Feature extraction and k-NN model training completed. Data saved for future runs.")
else:
    print("Features already extracted. Loading saved files...")
    image_features = pickle.load(open(features_file, "rb"))
    img_files = pickle.load(open(images_file, "rb"))
    knn = pickle.load(open(knn_model_file, "rb"))

print("Model Ready! You can now use the test or web application.")
