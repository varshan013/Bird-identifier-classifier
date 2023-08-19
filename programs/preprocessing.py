import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = cv2.resize(image, image_size)  # Resize the image
    image = img_to_array(image)  # Convert image to array
    image /= 255.0  # Normalize pixel values between 0 and 1
    return image

def load_dataset(dataset_path, image_size):
    image_data = []
    labels = []

    bird_species = sorted(os.listdir(dataset_path))
    for i, bird_species_name in enumerate(bird_species):
        bird_species_path = os.path.join(dataset_path, bird_species_name)
        for image_file in os.listdir(bird_species_path):
            image_path = os.path.join(bird_species_path, image_file)

            try:
                image = preprocess_image(image_path, image_size)
                image_data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(e)

    image_data = np.array(image_data, dtype="float32")
    labels = np.array(labels)

    return image_data, labels

# Define the path to your dataset folder
dataset_path = "dataset/training_set"
image_size = (224, 224)  # Adjust the image size according to your needs

# Load the dataset
image_data, labels = load_dataset(dataset_path, image_size)

# Save the preprocessed data and labels to files
np.save("data\preprocessed_data.npy", image_data)
np.save("data\preprocessed_labels.npy", labels)
