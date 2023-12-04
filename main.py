import cv2
import numpy as np
import pickle
import argparse
from utils import extractLbpFeatures, extractOrbFeatures, featureFusion  # Assuming these are in utils.py

def loadModel(modelPath):
    with open(modelPath, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocessImage(imagePath, faceCascade):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Assuming the first detected face is the one to be used
    x, y, w, h = faces[0]
    faceImage = image[y:y+h, x:x+w]

    # Extract features
    lbpFeatures = extractLbpFeatures(faceImage)
    orbFeatures = extractOrbFeatures(faceImage)

    # If using a combined model, concatenate the features
    combinedFeatures = np.concatenate([lbpFeatures, orbFeatures])

    return combinedFeatures

def main(imagePath):
    # Load the trained model
    modelPath = 'svmCombinedModel.pkl'
    model = loadModel(modelPath)

    # Load the Haar Cascade for face detection
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # Preprocess the image
    features = preprocessImage(imagePath, faceCascade)

    # Reshape features for single sample prediction
    features = features.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    # Output the result
    print("Predicted Emotion:", prediction[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('imagePath', type=str, help='Path to the image file')
    args = parser.parse_args()
    main(args.imagePath)
