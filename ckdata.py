import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib

'''
CSV Format: emotion number, pixels, usage

- Emotions/Expressions are defined as determined index below:

    0 : Anger
    1 : Disgust
    2 : Fear
    3 : Happiness
    4 : Sadness
    5 : Surprise
    6 : Neutral
    7 : Contempt

- Pixels contains 2304 pixels (48x48) each row

- Usage is determined as Training(80%) / PublicTest(10%) / PrivateTest(10%)
'''

# load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat')

# Load your CSV file
ckCSV = pd.read_csv("ckextended.csv")

for i in range(5):
    pixels = ckCSV.iloc[i]['pixels']
    pixels = list(map(int, pixels.split()))
    image = np.array(pixels).reshape(48, 48).astype(np.uint8)  # Adjust the shape as per your dataset
    # image = np.stack((image,)*3, axis=-1)

    # Detect faces in the image
    faces = detector(image)

    for face in faces:
        landmarks = predictor(image, face)

        # Extracting the facial region based on landmarks
        x = [landmarks.part(n).x for n in range(0, 27)]
        y = [landmarks.part(n).y for n in range(0, 27)]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        # Crop to the facial region
        face_region = image[y_min:y_max, x_min:x_max]

        # Display the cropped facial region
        plt.imshow(face_region, cmap='gray')
        plt.title(f"Face Region in Image {i+1}")
        plt.axis('off')
        plt.show()




