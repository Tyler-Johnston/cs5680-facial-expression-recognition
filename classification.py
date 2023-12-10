import argparse
from utils import processEmotionImages, featureFusion, displayResults
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from collections import Counter

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a facial emotion recognition model.')
parser.add_argument('--no_smote', action='store_true', help='Run without using SMOTE/UnderSampling for balancing the dataset')
parser.add_argument('--all_emotions', action='store_true', help='Use all emotions in the dataset')
args = parser.parse_args()

# Load and process the dataset
baseFolder = 'datasets/CK+'
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] if args.all_emotions else ["happiness", "neutral", "surprise"]
lbpFeatures, orbFeatures, yLabels, emotionImageCount = processEmotionImages(baseFolder, emotions, not args.no_smote)
combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K=100, C=1) # K = 100 for scaling factor spoke of in the paper. C = 1 is a constant to ensure there isn't a division by zero

def trainAndEvaluate(features, labels, featureType, useSmote):
    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    if useSmote:
        classDist = Counter(yLabels)
        maxSamples = max(classDist.values())
        minSamples = min(classDist.values())
        overStrategy = {k: maxSamples for k in classDist.keys()}
        underStrategy = {k: minSamples for k in classDist.keys()}
        model = Pipeline([
            ('over', SMOTE(sampling_strategy=overStrategy, random_state=42)),
            ('under', RandomUnderSampler(sampling_strategy=underStrategy, random_state=42)),
            ('model', SVC(kernel='linear', class_weight='balanced', random_state=42))
        ])
    else:
        model = SVC(kernel='linear', class_weight='balanced', random_state=42)

    model.fit(xTrain, yTrain)
    with open(f'{featureType}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    yPred = model.predict(xTest)
    accuracyScore = accuracy_score(yTest, yPred)
    classificationReport = classification_report(yTest, yPred, zero_division=0)
    confusionMatrix = confusion_matrix(yTest, yPred)
    classAccuracies = confusionMatrix.diagonal() / confusionMatrix.sum(axis=1)
    displayResults(confusionMatrix, classAccuracies, accuracyScore, classificationReport, emotions, featureType, emotionImageCount)

# Train and evaluate models
trainAndEvaluate(lbpFeatures, yLabels, 'LBP', not args.no_smote)
trainAndEvaluate(orbFeatures, yLabels, 'ORB', not args.no_smote)
trainAndEvaluate(combinedFeatures, yLabels, 'Combined', not args.no_smote)