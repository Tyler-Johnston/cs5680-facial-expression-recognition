from utils import processEmotionImages, featureFusion, plot_confusion_matrix_and_accuracies
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import compute_class_weight
import numpy as np
import pickle
from collections import Counter

# Load and process the dataset
baseFolder = 'datasets/CK+'
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
lbpFeatures, orbFeatures, yLabels = processEmotionImages(baseFolder, emotions)
combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K=100, C=1)

def create_and_evaluate_model(features, labels, feature_type):
    # Calculate the class distribution
    class_dist = Counter(labels)
    
    # Find the maximum number of samples in any class
    max_samples = max(class_dist.values())
    
    # Define the over-sampling strategy
    # Increase minority class counts up to the count of the majority class
    over_strategy = {label: min(count, max_samples) for label, count in class_dist.items() if label != 'neutral'}
    
    # Define the under-sampling strategy
    # Reduce majority class counts down to the count of the smallest minority class
    # (This could be adjusted to a different target as needed)
    min_samples = min(class_dist.values())
    under_strategy = {label: min_samples for label in class_dist.keys()}
    

    # Create a pipeline with the new strategies
    model_pipeline = Pipeline([
        ('over', SMOTE(sampling_strategy=over_strategy, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
        ('model', SVC(kernel='linear', class_weight='balanced', random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Save the model
    with open(f'{feature_type}_model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

    # Evaluate on test data
    y_pred = model_pipeline.predict(X_test)
    print(f"\n{feature_type} Model - Test Set Evaluation")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"{feature_type} Accuracy:", accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plot_confusion_matrix_and_accuracies(conf_matrix, class_accuracies, emotions, f"{feature_type} Confusion Matrix")

# Create and evaluate models
create_and_evaluate_model(lbpFeatures, yLabels, 'LBP')
create_and_evaluate_model(orbFeatures, yLabels, 'ORB')
create_and_evaluate_model(combinedFeatures, yLabels, 'Combined')
