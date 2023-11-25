import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# Load Haar Cascade for face detection
# This returns a list of rectangles where faces were detected. 
# Each rectangle is represented by the coordinates (x, y) and the width and height (w, h)
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
        face_region = image[y:y+h, x:x+w]
        
        # Display the extracted face region
        plt.imshow(face_region, cmap='gray')
        plt.title(f"Extracted Face from Image {i+1}")
        plt.axis('off')
        plt.show()