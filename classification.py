# from utils import processEmotionImages, featureFusion
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score
# import numpy as np
# import pickle

# # CK+ Dataset Specific Information
# baseFolder = 'datasets/CK+'
# emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# # Get LBP, ORB, and Fused Features
# allFeatures = processEmotionImages(baseFolder, emotions)

# # Preparing the dataset for training
# xLbp = []
# xOrb = []
# xCombined = featureFusion(allFeatures)
# yLabels = []

# for emotion, features in allFeatures.items():
#     for featureVector in features['LBP']:
#         xLbp.append(featureVector)
#         yLabels.append(emotion)

#     for featureVector in features['ORB']:
#         xOrb.append(featureVector.flatten())

# print("SHAPES: ")
# print(xLbp.shape())
# print(xOrb.shape())
# print(xCombined.shape())

# # Split the dataset into training and testing sets for LBP
# xLbpTrain, xLbpTest, yLbpTrain, yLbpTest = train_test_split(xLbp, yLabels, test_size=0.2, random_state=95)

# # Split the dataset into training and testing sets for ORB
# xOrbTrain, xOrbTest, yOrbTrain, yOrbTest = train_test_split(xOrb, yLabels, test_size=0.2, random_state=95)

# # Split the dataset into training and testing sets for combined features
# xCombinedTrain, xCombinedTest, yCombinedTrain, yCombinedTest = train_test_split(xCombined, yLabels, test_size=0.2, random_state=95)

# # Training the SVM Classifier for LBP features
# svmLbp = SVC(kernel='linear', random_state=95)
# svmLbp.fit(xLbpTrain, yLbpTrain)

# # Training the SVM Classifier for ORB features
# svmOrb = SVC(kernel='linear', random_state=95)
# svmOrb.fit(xOrbTrain, yOrbTrain)

# # Training the SVM Classifier for combined features
# svmCombined = SVC(kernel='linear', random_state=95)
# svmCombined.fit(xCombinedTrain, yCombinedTrain)

# # Save the models to disk
# with open('svmLbpModel.pkl', 'wb') as f:
#     pickle.dump(svmLbp, f)
# with open('svmOrbModel.pkl', 'wb') as f:
#     pickle.dump(svmOrb, f)
# with open('svmCombinedModel.pkl', 'wb') as f:
#     pickle.dump(svmCombined, f)

# # Prediction and evaluation for LBP
# yPredLbp = svmLbp.predict(xLbpTest)
# print("LBP Classification Report:")
# print(classification_report(yLbpTest, yPredLbp, zero_division=0))
# print("LBP Accuracy:", accuracy_score(yLbpTest, yPredLbp))

# # Prediction and evaluation for ORB
# yPredOrb = svmOrb.predict(xOrbTest)
# print("ORB Classification Report:")
# print(classification_report(yOrbTest, yPredOrb, zero_division=0))
# print("ORB Accuracy:", accuracy_score(yOrbTest, yPredOrb))

# # Prediction and evaluation for Combined Features
# yPredCombined = svmCombined.predict(xCombinedTest)
# print("Combined Feature Classification Report:")
# print(classification_report(yCombinedTest, yPredCombined, zero_division=0))
# print("Combined Feature Accuracy:", accuracy_score(yCombinedTest, yPredCombined))
from utils import processEmotionImages, featureFusion
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pickle

# CK+ Dataset Specific Information
baseFolder = 'datasets/CK+'
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Get LBP, ORB, and Fused Features
allFeatures = processEmotionImages(baseFolder, emotions)
xCombined = featureFusion(allFeatures)

# Preparing the dataset for training using NumPy arrays
xLbp = np.empty((0, len(allFeatures[emotions[0]]['LBP'][0])), float)
xOrb = np.empty((0, len(allFeatures[emotions[0]]['ORB'][0])), float)
yLabels = []

for emotion, features in allFeatures.items():
    for featureVector in features['LBP']:
        xLbp = np.vstack([xLbp, featureVector])
        yLabels.append(emotion)

    for featureVector in features['ORB']:
        xOrb = np.vstack([xOrb, featureVector.flatten()])

yLabels = np.array(yLabels)

# Split the dataset into training and testing sets for LBP
xLbpTrain, xLbpTest, yLbpTrain, yLbpTest = train_test_split(xLbp, yLabels, test_size=0.2, random_state=95)

# Split the dataset into training and testing sets for ORB
xOrbTrain, xOrbTest, yOrbTrain, yOrbTest = train_test_split(xOrb, yLabels, test_size=0.2, random_state=95)

# Split the dataset into training and testing sets for combined features
xCombinedTrain, xCombinedTest, yCombinedTrain, yCombinedTest = train_test_split(np.array(xCombined), yLabels, test_size=0.2, random_state=95)

# Training the SVM Classifier for LBP features
svmLbp = SVC(kernel='linear', random_state=95)
svmLbp.fit(xLbpTrain, yLbpTrain)

# Training the SVM Classifier for ORB features
svmOrb = SVC(kernel='linear', random_state=95)
svmOrb.fit(xOrbTrain, yOrbTrain)

# Training the SVM Classifier for combined features
svmCombined = SVC(kernel='linear', random_state=95)
svmCombined.fit(xCombinedTrain, yCombinedTrain)

# Save the models to disk
with open('svmLbpModel.pkl', 'wb') as f:
    pickle.dump(svmLbp, f)
with open('svmOrbModel.pkl', 'wb') as f:
    pickle.dump(svmOrb, f)
with open('svmCombinedModel.pkl', 'wb') as f:
    pickle.dump(svmCombined, f)

# Prediction and evaluation for LBP
yPredLbp = svmLbp.predict(xLbpTest)
print("LBP Classification Report:")
print(classification_report(yLbpTest, yPredLbp, zero_division=0))
print("LBP Accuracy:", accuracy_score(yLbpTest, yPredLbp))

# Prediction and evaluation for ORB
yPredOrb = svmOrb.predict(xOrbTest)
print("ORB Classification Report:")
print(classification_report(yOrbTest, yPredOrb, zero_division=0))
print("ORB Accuracy:", accuracy_score(yOrbTest, yPredOrb))

# Prediction and evaluation for Combined Features
yPredCombined = svmCombined.predict(xCombinedTest)
print("Combined Feature Classification Report:")
print(classification_report(yCombinedTest, yPredCombined, zero_division=0))
print("Combined Feature Accuracy:", accuracy_score(yCombinedTest, yPredCombined))
