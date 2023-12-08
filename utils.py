import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import seaborn as sns

# Function to extract LBP features
def extractLbpFeatures(faceImage, displayLbp=False):
    radius = 1  # Radius of the LBP operation
    nPoints = 8 * radius  # Number of points to consider
    lbp = local_binary_pattern(faceImage, nPoints, radius, method="uniform")
    nBins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=nBins, range=(0, nBins), density=True)
    
    # Display the LBP image
    if displayLbp:
        plt.imshow(lbp, cmap='gray')
        plt.title("LBP Image")
        plt.axis('off')
        plt.show()
    
    return hist

def extractOrbFeatures(faceImage, maxKeypoints=500, descriptorSize=32, displayOrb=False):
    orb = cv2.ORB_create(nfeatures=maxKeypoints)
    keypoints, descriptors = orb.detectAndCompute(faceImage, None)

    # Draw keypoints on the image
    if keypoints and displayOrb:
        keypointImage = cv2.drawKeypoints(faceImage, keypoints, None, color=(0, 255, 0), flags=0)
        plt.imshow(keypointImage)
        plt.title("ORB Keypoints")
        plt.axis('off')
        plt.show()
    
    if descriptors is not None:
        # Flatten and pad the descriptor array
        flattened_descriptors = descriptors.flatten()
        padded_length = maxKeypoints * descriptorSize
        padded_descriptors = np.zeros(padded_length)
        padded_descriptors[:len(flattened_descriptors)] = flattened_descriptors
        return padded_descriptors
    else:
        # Return a zero vector if no keypoints are detected
        return np.zeros(maxKeypoints * descriptorSize)

def processEmotionImages(baseFolder, emotions):
    # Load the Haar Cascade for face detection
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    # Initialize empty lists to hold the LBP and ORB features
    lbpFeaturesList = []
    orbFeaturesList = []
    # Initialize the emotional state "ground truth" list
    yLabels = []

    # For each image in the CK+ dataset, obtain the LBP and ORB features alongwith the ground truth emotion
    for emotion in emotions:
        emotionFolder = os.path.join(baseFolder, emotion)

        for filename in os.listdir(emotionFolder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                imagePath = os.path.join(emotionFolder, filename)
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                
                # Detect faces in the image
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:
                    continue  # Skip images where no face is detected
                x, y, w, h = faces[0]  # Use the first detected face
                faceImage = image[y:y+h, x:x+w]  # Extract the face region

                # Extract features
                lbpFeatures = extractLbpFeatures(faceImage)
                orbFeatures = extractOrbFeatures(faceImage)

                if lbpFeatures is not None:
                    lbpFeaturesList.append(lbpFeatures)
                if orbFeatures is not None:
                    orbFeaturesList.append(orbFeatures)
                yLabels.append(emotion)

    return lbpFeaturesList, orbFeaturesList, yLabels

def zScoreNormalization(features, K, C):
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    normalizedFeatures = K * ((features - mu) / (sigma + C))
    return normalizedFeatures

def featureFusion(lbpFeatures, orbFeatures, K, C, singleAxis=True):
    # Normalize each feature set
    lbpNormalized = zScoreNormalization(lbpFeatures, K, C)
    orbNormalized = zScoreNormalization(orbFeatures, K, C)

    # Concatenate the normalized features
    if singleAxis:
        fusedFeatures = np.concatenate((lbpNormalized, orbNormalized), axis=1)
    else:
        fusedFeatures = np.concatenate((lbpNormalized, orbNormalized))

    return fusedFeatures

def plot_confusion_matrix_and_accuracies(conf_matrix, class_accuracies, emotions, title):
    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Show the plot
    plt.show()

    # Print class accuracies
    print(f"\n{title} Accuracies:")
    for i, accuracy in enumerate(class_accuracies):
        print(f"{emotions[i]}: {accuracy:.2f}")