import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from utils import extractLbpFeatures, extractOrbFeatures, featureFusion

def loadModel(modelPath):
    with open(modelPath, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocessImage(imagePath, faceCascade, featureType):
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

    # Determine which features to use
    if featureType == 'combined':
        combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K=100, C=1, singleAxis=False)
        return combinedFeatures
    elif featureType == 'orb':
        return orbFeatures
    else:
        return lbpFeatures

def main(imagePath, modelPath):
    # Load the Haar Cascade for face detection
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # Determine feature type based on model filename
    featureType = 'combined' if 'Combined' in modelPath else 'orb' if 'ORB' in modelPath else 'lbp'
    print(featureType)

    # Load model and preprocess the image
    model = loadModel(modelPath)
    features = preprocessImage(imagePath, faceCascade, featureType)

    # Reshape features for single sample prediction
    features = features.reshape(1, -1)

    # Make a prediction and get decision function scores
    prediction = model.predict(features)
    decisionScores = model.decision_function(features)

    # Get the indices of the top predictions based on decision scores
    topIndices = np.argsort(decisionScores[0])[-4:]

    # Get the decision scores for the top predictions
    topScores = decisionScores[0][topIndices]

    # Get the emotion labels for the top predictions
    topEmotions = model.classes_[topIndices]

    # Display the image and the top predictions with their decision scores
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"Predicted Emotion: {prediction[0]}")
    for i, (emotion, score) in enumerate(zip(topEmotions, topScores)):
        plt.text(10, 30 + 30*i, f"{emotion}: {score:.2f}", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.show()

# USAGE: python3 main.py path_to_image path_to_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('imagePath', type=str, help='Path to the image file')
    parser.add_argument('modelPath', type=str, help='Path to the model file')
    args = parser.parse_args()
    main(args.imagePath, args.modelPath)