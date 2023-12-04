import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from utils import extractLbpFeatures, extractOrbFeatures

def loadModel(modelPath):
    with open(modelPath, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocessImage(imagePath, faceCascade, useOrb=False, useCombined=False):
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
    orbFeatures = extractOrbFeatures(faceImage) if useOrb or useCombined else None

    # Determine which features to use
    if useCombined:
        combinedFeatures = np.concatenate([lbpFeatures, orbFeatures])
        return combinedFeatures
    elif useOrb:
        return orbFeatures
    else:
        return lbpFeatures

def main(imagePath, modelType):
    # Select model based on input
    modelPaths = {
        'lbp': 'svmLbpModel.pkl',
        'orb': 'svmOrbModel.pkl',
        'combined': 'svmCombinedModel.pkl'
    }
    modelPath = modelPaths[modelType]
    model = loadModel(modelPath)

    # Load the Haar Cascade for face detection
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # Preprocess the image
    features = preprocessImage(imagePath, faceCascade, useOrb=(modelType=='orb'), useCombined=(modelType=='combined'))

    # Reshape features for single sample prediction
    features = features.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    # Read the image and plot it
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"Predicted Emotion: {prediction[0]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('imagePath', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, choices=['lbp', 'orb', 'combined'], default='combined', help='Model type (lbp, orb, or combined)')
    args = parser.parse_args()
    main(args.imagePath, args.model)
