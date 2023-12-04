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

## Phase 2: Feature Fusion and SVM Classification

I realized I was using a low quality version of the CK+ dataset, and I modified my code to accept this new format structure. This corrected the ORB issues I was having in phase 1. In addition to this, I implemented these new features:

### Feature Fusion
In this phase, we developed a method to combine the Local Binary Patterns (LBP) and Oriented FAST and Rotated BRIEF (ORB) features, known as feature fusion. This approach aims to leverage the strengths of both feature sets for improved facial expression classification. The fusion process involves concatenating LBP and ORB feature vectors for each image.

```python
def FeatureFusion(allFeatures):
    X_combined = []
    y_combined = []
    
    for emotion, features in allFeatures.items():
        for lbp_feature, orb_feature in zip(features['LBP'], features['ORB']):
            # Flatten the ORB feature and concatenate it with the LBP feature
            combined_feature = np.concatenate([lbp_feature, orb_feature.flatten()])
            X_combined.append(combined_feature)
            y_combined.append(emotion)
    return X_combined, y_combined
```

### SVM Classifier

I also employed a Support Vector Machine (SVM) for emotion classification, using the LBP, ORB, and the fused features. SVMs are known for their effectiveness in high-dimensional spaces, making them suitable for our fused feature set. We trained separate SVM models for the LBP, ORB, and combined feature sets, providing a comprehensive analysis of each method's effectiveness. I implemented the SVM classifier using the sklearn library.

```python
# Preparing the dataset for training
X_lbp = []
X_orb = []
y = []

for emotion, features in allFeatures.items():
    for feature_vector in features['LBP']:
        X_lbp.append(feature_vector)
        y.append(emotion)  # Use numerical labels for emotions if necessary

    for feature_vector in features['ORB']:
        # Flatten the feature vector due to it being multi-dimensional
        X_orb.append(feature_vector.flatten())

# Split the dataset into training and testing sets for LBP and ORB
X_lbp_train, X_lbp_test, y_train, y_test = train_test_split(X_lbp, y, test_size=0.2, random_state=42)
X_orb_train, X_orb_test, _, _ = train_test_split(X_orb, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Training the SVM Classifier for LBP features
svm_lbp = SVC(kernel='linear', random_state=42)
svm_lbp.fit(X_lbp_train, y_train)

# Training the SVM Classifier for ORB features
svm_orb = SVC(kernel='linear', random_state=42)
svm_orb.fit(X_orb_train, y_train)

# Training the SVM Classifier for combined features
svm_combined = SVC(kernel='linear', random_state=42)
svm_combined.fit(X_train, y_train)
```

As of right now, the LBP and ORB and the combined are about 66% accurate. The dataset has a lot of neutral faces but not a lot of angry, fearful, sadness, etc faces, which is causing a bias in the prediction model. I will need to fix this to improve the results in the next phase.