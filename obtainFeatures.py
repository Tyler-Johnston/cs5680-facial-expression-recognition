import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load the Haar Cascade for face detection
cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Function to extract LBP features
def extractLbpFeatures(faceImage):
    radius = 1  # Radius of the LBP operation
    nPoints = 8 * radius  # Number of points to consider
    lbp = local_binary_pattern(faceImage, nPoints, radius, method="uniform")
    nBins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=nBins, range=(0, nBins), density=True)
    
    # # Display the LBP image
    # plt.imshow(lbp, cmap='gray')
    # plt.title("LBP Image")
    # plt.axis('off')
    # plt.show()
    
    return hist

def extractOrbFeatures(faceImage, maxKeypoints=500, descriptorSize=32):
    orb = cv2.ORB_create(nfeatures=maxKeypoints)
    keypoints, descriptors = orb.detectAndCompute(faceImage, None)
    
    if descriptors is not None:
        # Flatten and pad the descriptor array
        flattened_descriptors = descriptors.flatten()
        padded_length = maxKeypoints * descriptorSize
        padded_descriptors = np.zeros(padded_length)
        padded_descriptors[:len(flattened_descriptors)] = flattened_descriptors
        return padded_descriptors
    else:
        # Return a zero vector if no keypoints are detected
        return np.zeros(max_keypoints * descriptor_size)


# # Function to extract ORB features
# def extractOrbFeatures(faceImage):
#     orb = cv2.ORB_create()
#     keypoints, descriptors = orb.detectAndCompute(faceImage, None)
    
#     # # Draw keypoints on the image
#     # if keypoints:
#     #     keypointImage = cv2.drawKeypoints(faceImage, keypoints, None, color=(0, 255, 0), flags=0)
#     #     plt.imshow(keypointImage)
#     #     plt.title("ORB Keypoints")
#     #     plt.axis('off')
#     #     plt.show()
    
#     return keypoints, descriptors

# Main function to process images in each emotion folder
def processEmotionImages(baseFolder):
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    # emotions = ['anger']
    features = {}

    for emotion in emotions:
        emotionFolder = os.path.join(baseFolder, emotion)
        lbpFeaturesList = []
        orbFeaturesList = []

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

                lbpFeaturesList.append(lbpFeatures)
                if orbFeatures is not None:
                    orbFeaturesList.append(orbFeatures)

        features[emotion] = {
            'LBP': lbpFeaturesList,
            'ORB': orbFeaturesList
        }

    return features