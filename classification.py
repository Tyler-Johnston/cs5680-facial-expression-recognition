from utils import processEmotionImages, FeatureFusion
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# CK+ Dataset Specific Information
baseFolder = 'datasets/CK+'
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Get LBP, ORB, and Fused Features
allFeatures = processEmotionImages(baseFolder, emotions)
X_combined, y_combined = FeatureFusion(allFeatures)

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

# Save the models to disk
with open('svm_lbp_model.pkl', 'wb') as f:
    pickle.dump(svm_lbp, f)
with open('svm_orb_model.pkl', 'wb') as f:
    pickle.dump(svm_orb, f)
with open('svm_combined_model.pkl', 'wb') as f:
    pickle.dump(svm_combined, f)

# Prediction and evaluation for LBP
y_pred_lbp = svm_lbp.predict(X_lbp_test)
print("LBP Classification Report:")
print(classification_report(y_test, y_pred_lbp))
print("LBP Accuracy:", accuracy_score(y_test, y_pred_lbp))

y_pred_orb = svm_orb.predict(X_orb_test)
print("ORB Classification Report:")
print(classification_report(y_test, y_pred_orb))
print("ORB Accuracy:", accuracy_score(y_test, y_pred_orb))

y_pred_combined = svm_combined.predict(X_test)
print("Combined Feature Classification Report:")
print(classification_report(y_test, y_pred_combined))
print("Combined Feature Accuracy:", accuracy_score(y_test, y_pred_combined))