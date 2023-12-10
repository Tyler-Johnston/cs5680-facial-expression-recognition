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
# emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
emotions = ["happiness", "neutral", "surprise"] # per Dr. Qi's instructions, look at 100 neutral images, 83 surprise images, and 69 happiness images to train the model
lbpFeatures, orbFeatures, yLabels = processEmotionImages(baseFolder, emotions)
combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K=100, C=1)

def trainAndEvaluate(features, labels, featureType):
    # Assuming yLabels contains your class labels
    classDist = Counter(yLabels)

    # Find the maximum and minimum number of samples in any class
    maxSamples = max(classDist.values())
    minSamples = min(classDist.values())

    # Define the over-sampling strategy: Oversample all classes to the count of the most populous class
    overStrategy = {k: maxSamples for k in classDist.keys()}

    # Define the under-sampling strategy: Under-sample all classes but the least populous class
    underStrategy = {k: minSamples for k in classDist.keys()}

    # Create a pipeline with the new strategies
    modelPipeline = Pipeline([
        ('over', SMOTE(sampling_strategy=overStrategy, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=underStrategy, random_state=42)),
        ('model', SVC(kernel='linear', class_weight='balanced', random_state=42))
    ])

    # Split your data into training and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(features, yLabels, test_size=0.2, random_state=42)

    # Fit the model
    modelPipeline.fit(xTrain, yTrain)

    # Save the model
    with open(f'{featureType}_model.pkl', 'wb') as f:
        pickle.dump(modelPipeline, f)

    # Evaluate on test data
    yPred = modelPipeline.predict(xTest)
    print(f"\n{featureType} Model - Test Set Evaluation")
    print(classification_report(yTest, yPred, zero_division=0))
    print(f"{featureType} Accuracy:", accuracy_score(yTest, yPred))
    conf_matrix = confusion_matrix(yTest, yPred)
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plot_confusion_matrix_and_accuracies(conf_matrix, class_accuracies, emotions, f"{featureType} Confusion Matrix")

# train and evaluate on LBP, ORB, and the Combined Feature Sets
trainAndEvaluate(lbpFeatures, yLabels, 'LBP')
trainAndEvaluate(orbFeatures, yLabels, 'ORB')
trainAndEvaluate(combinedFeatures, yLabels, 'Combined')