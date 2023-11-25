# Facial Expression Recognition Project

## Overview
This project focuses on implementing a facial expression recognition algorithm based on Local Binary Patterns (LBP) and Oriented FAST and Rotated BRIEF (ORB) features. The goal is to accurately classify facial expressions from image data, extracted from a provided CSV dataset.

## Phase I: Face Extraction; LBP and ORB Feature Extraction on CK Dataset

During phase 1, the face extraction, LBP, and ORB features are obtained using the CK Dataset inititially. The ORB features are currently not working as intended, and because this was during the thanksgiving break, only the groundwork and inital setup code was completed. However, this is more work completed than initally anticipated during the project proposal. The proposal only anticiapted the face-extraction portion to be completed during the week, so I am currently ahead of schedule. 

This was done on the CK dataset, which is a CSV file in an emotion number, pixels, and usage format. Here is a description of what those represent:

    - Emotions/Expressions are defined as determined index below:

        0 : Anger
        1 : Disgust
        2 : Fear
        3 : Happiness
        4 : Sadness
        5 : Surprise
        6 : Neutral
        7 : Contempt

    - Pixels contains 2304 pixels (48x48) each for row

    - Usage is determined as Training(80%) / PublicTest(10%) / PrivateTest(10%)

Thus the initial images are obtained from converting the grayscale 0 - 255 pixels to images, extracting the faces from them, and obtaining the LBP and ORB feature sets. In future phases, the emotion will be calculated and cross-checked with the truth-emotion values in this CSV 

### Face Extraction

- **Purpose**: Accurate facial expression recognition requires isolating the face from the rest of the image. This ensures that the subsequent feature extraction focuses solely on the facial features relevant for emotion classification.
- **Implementation**: Face extraction is performed using OpenCV's Haar Cascade Classifier. The classifier detects faces in each image and returns coordinates for the facial region. This region is then cropped from the image for further processing.


    ```python
    # Load the Haar Cascade for face detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect the faces in the image
    faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
            # Extract the face region from the image
            extractedFace = image[y:y+h, x:x+w]

    ```

### Local Binary Patterns (LBP)
- **Purpose**: LBP is used for capturing fine-grained texture information from facial images, which is crucial for recognizing subtle differences in facial expressions.
- **Implementation**: LBP features are extracted from each detected face region within the image. The implementation involves converting the grayscale face region into a local binary pattern representation and then computing a histogram of these patterns.

    ```python
    def extractLbpFeatures(image):
        # Extracts LBP features from a given image
        # Parameters for LBP are set to a radius of 1 and 8 points
        lbpImage = local_binary_pattern(image, 8, 1, method='uniform')
        lbpHist, _ = np.histogram(lbpImage.ravel(), bins=int(lbpImage.max()) + 1, range=(0, int(lbpImage.max()) + 1), density=True)
        return lbpHist

    # ... [extracted face defined previously] ...
    lbpFeatures = extractLbpFeatures(extractedFace)
    ```

### Oriented FAST and Rotated BRIEF (ORB)
- **Purpose**: ORB is utilized for efficient keypoint detection and description, enhancing the feature set used for facial expression classification.
- **Implementation**: ORB features are extracted by detecting keypoints in the facial region and computing their descriptors.

    ```python
        def extractOrbFeatures(image):
        # Extracts ORB features from a given image
        # ORB parameters are the default settings provided by OpenCV
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return descriptors

    # ... [extracted face defined previously] ...
    orbFeatures = extractOrbFeatures(extractedFace)
    ```

