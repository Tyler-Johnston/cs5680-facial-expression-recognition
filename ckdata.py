import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

'''
CK Data CSV Format: emotion number, pixels, usage

- Emotions/Expressions are defined as determined index below:

    0 : Anger
    1 : Disgust
    2 : Fear
    3 : Happiness
    4 : Sadness
    5 : Surprise
    6 : Neutral
    7 : Contempt

- Pixels contains 2304 pixels (48x48) for each row

- Usage is determined as Training(80%) / PublicTest(10%) / PrivateTest(10%)
'''

# LBP feature extraction function
# Each array in the LBP feature set represents the histogram of LBP patterns for a given face region,
def extractLbpFeatures(image):
    radius = 1  # Radius of the circle
    numberOfPoints = 8 * radius  # Number of points to consider around the circle
    lbpImage = local_binary_pattern(image, numberOfPoints, radius, method='uniform')
    numberOfBins = int(lbpImage.max() + 1)
    lbpHistogram, _ = np.histogram(lbpImage.ravel(), bins=numberOfBins, range=(0, numberOfBins), density=True)

    # # Display the LBP image
    # plt.imshow(lbpImage, cmap='gray')
    # plt.title("Local Binary Pattern")
    # plt.axis('off')
    # plt.show()

    return lbpHistogram

# ORB feature extraction function
def extractOrbFeatures(image):
    # orb = cv2.ORB_create(nfeatures=10, edgeThreshold=10)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Draw keypoints on the image
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

    # # Display the image using matplotlib
    # plt.imshow(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB))
    # plt.title("Image with ORB Keypoints")
    # plt.axis('off')
    # plt.show()
    return descriptors

# Load the Haar Cascade for face detection
# The paper did not use Haar Cascade, but it noted that this was an option. 
# I had issues using the dlib library that the actually used, and this alternative is working as expected
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the CSV file
ckCSV = pd.read_csv("datasets/ckextended.csv")

# Initialize lists to store LBP and ORB features
lbpFeaturesList = []
orbFeaturesList = []

# for i in range(len(ckCSV)):
for i in range(5):
    pixels = ckCSV.iloc[i]['pixels']
    pixels = list(map(int, pixels.split()))
    image = np.array(pixels).reshape(48, 48).astype(np.uint8)

    faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the image
        extractedFace = image[y:y+h, x:x+w]

        # # Display the extracted face region
        # plt.imshow(extractedFace, cmap='gray')
        # plt.title(f"Extracted Face from Image {i+1}")
        # plt.axis('off')
        # plt.show()

        # Extract LBP and ORB features
        lbpFeatures = extractLbpFeatures(extractedFace)
        orbFeatures = extractOrbFeatures(extractedFace)

        # Store the features in the lists
        if lbpFeatures is not None:
            lbpFeaturesList.append(lbpFeatures)
        if orbFeatures is not None:
            orbFeaturesList.append(orbFeatures)

print("LBP: ", lbpFeaturesList)
print("ORB: ", orbFeaturesList)