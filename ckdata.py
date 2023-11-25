import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# LBP feature extraction function
def extractLbpFeatures(image):
    radius = 1  # Radius of the circle
    nPoints = 8 * radius  # Number of points to consider around the circle
    lbpImage = local_binary_pattern(image, nPoints, radius, method='uniform')
    nBins = int(lbpImage.max() + 1)
    lbpHist, _ = np.histogram(lbpImage.ravel(), bins=nBins, range=(0, nBins), density=True)
    return lbpHist

# ORB feature extraction function
def extractOrbFeatures(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

# Load the Haar Cascade for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your CSV file
ckCSV = pd.read_csv("ckextended.csv")

for i in range(5):
    pixels = ckCSV.iloc[i]['pixels']
    pixels = list(map(int, pixels.split()))
    image = np.array(pixels).reshape(48, 48).astype(np.uint8)

    faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the image
        extractedFace = image[y:y+h, x:x+w]

        # Display the extracted face region
        plt.imshow(extractedFace, cmap='gray')
        plt.title(f"Extracted Face from Image {i+1}")
        plt.axis('off')
        plt.show()

        # Extract LBP and ORB features
        lbpFeatures = extractLbpFeatures(extractedFace)
        orbFeatures = extractOrbFeatures(extractedFace)

        # You can now use lbpFeatures and orbFeatures for further analysis
        print(f"LBP Features for Image {i+1}: {lbpFeatures}")
        print(f"ORB Features for Image {i+1}: {orbFeatures}")
