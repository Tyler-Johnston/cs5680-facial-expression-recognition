import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the model

# Function to load your dataset
def load_dataset():
    # Load your dataset here (LBP, ORB, or fused features)
    # For example:
    # features = np.load('features.npy')
    # labels = np.load('labels.npy')
    # return features, labels
    pass

# Preprocess the data (if necessary)
def preprocess_data(features):
    # Implement any preprocessing steps like normalization here
    # return preprocessed_features
    pass

# Main function
def main():
    # Load and preprocess the dataset
    features, labels = load_dataset()
    features = preprocess_data(features)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define the SVM classifier
    clf = svm.SVC(kernel='linear')

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save the model
    joblib.dump(clf, 'svm_classifier.pkl')

if __name__ == "__main__":
    main()
