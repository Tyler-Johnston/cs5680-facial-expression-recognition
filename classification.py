from utils import processEmotionImages, featureFusion, plot_confusion_matrix_and_accuracies
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pickle

# CK+ Dataset Specific Information
baseFolder = 'datasets/CK+'
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Get LBP and ORB Features, alongwith the yLabels ground-truth
lbpFeatures, orbFeatures, yLabels = processEmotionImages(baseFolder, emotions)
K = 100  # Scaling factor as mentioned in the paper
C = 1    # A chosen constant to avoid division by zero in standard deviation
combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K, C)

# Split the dataset into training and testing sets for LBP
xLbpTrain, xLbpTest, yLbpTrain, yLbpTest = train_test_split(lbpFeatures, yLabels, test_size=0.2, random_state=95)

# Split the dataset into training and testing sets for ORB
xOrbTrain, xOrbTest, yOrbTrain, yOrbTest = train_test_split(orbFeatures, yLabels, test_size=0.2, random_state=95)

# Split the dataset into training and testing sets for combined features
xCombinedTrain, xCombinedTest, yCombinedTrain, yCombinedTest = train_test_split(combinedFeatures, yLabels, test_size=0.2, random_state=95)

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
confMatrixLbp = confusion_matrix(yLbpTest, yPredLbp)
classAccuraciesLbp = confMatrixLbp.diagonal() / confMatrixLbp.sum(axis=1)

# print("LBP Classification Report:")
# print(classification_report(yLbpTest, yPredLbp, zero_division=0))
# print("LBP Accuracy:", accuracy_score(yLbpTest, yPredLbp))

# Prediction and evaluation for ORB
yPredOrb = svmOrb.predict(xOrbTest)
confMatrixOrb = confusion_matrix(yOrbTest, yPredOrb)
classAccuraciesOrb = confMatrixOrb.diagonal() / confMatrixOrb.sum(axis=1)

# print("ORB Classification Report:")
# print(classification_report(yOrbTest, yPredOrb, zero_division=0))
# print("ORB Accuracy:", accuracy_score(yOrbTest, yPredOrb))

# Prediction and evaluation for Combined Features
yPredCombined = svmCombined.predict(xCombinedTest)
confMatrixCombined = confusion_matrix(yCombinedTest, yPredCombined)
classAccuraciesCombined = confMatrixCombined.diagonal() / confMatrixCombined.sum(axis=1)

# print("Combined Feature Classification Report:")
# print(classification_report(yCombinedTest, yPredCombined, zero_division=0))
# print("Combined Feature Accuracy:", accuracy_score(yCombinedTest, yPredCombined))

# For LBP Model
print("LBP Model Results:")
plot_confusion_matrix_and_accuracies(confMatrixLbp, classAccuraciesLbp, emotions, "LBP Confusion Matrix")

# For ORB Model
print("\nORB Model Results:")
plot_confusion_matrix_and_accuracies(confMatrixOrb, classAccuraciesOrb, emotions, "ORB Confusion Matrix")

# For Combined Model
print("\nCombined Model Results:")
plot_confusion_matrix_and_accuracies(confMatrixCombined, classAccuraciesCombined, emotions, "LBP+ORB Confusion Matrix")