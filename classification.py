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

# parse command-line arguments
parser = argparse.ArgumentParser(description='Run a facial emotion recognition analyzer')
parser.add_argument('--unbalance', action='store_true', help='Run without using SMOTE/UnderSampling for balancing the dataset')
parser.add_argument('--all_emotions', action='store_true', help='Use all emotions in the dataset')
parser.add_argument('--save_model', action='store_true', help='Save the trained model to disk')
args = parser.parse_args()

# load and process the dataset
baseFolder = 'datasets/CK+'
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] if args.all_emotions else ["happiness", "neutral", "surprise"]
lbpFeatures, orbFeatures, yLabels, emotionImageCount = processEmotionImages(baseFolder, emotions, not args.unbalance)
combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K=100, C=1) # K = 100 for scaling factor spoke of in the paper. C = 1 is a constant to ensure there isn't a division by zero

def trainAndEvaluate(features, labels, featureType, useRebalancing):
    '''
        - inputs: 1) features: an array of arrays the LBP, ORB, or Combined Feature set for each image
                  2) labels: an array containing the ground-truth emotion associated with each image in the feature set
                  3) featureType: a string describing which feature set is being processed
                  4) useRebalancing: a boolean describing if the model should use SMOTE/UnderSampling techniques
        - outputs: 1) the LBP, ORB, and Combined models, if specified by command-line arguments
                   2) the evaluation metrics (classification report, confusion matrices, class accuracies, feature set accuracy)
        - description: this trains the SVM given a feature set, saves the model if specified, and displays the metrics associated with it
    '''
    # split dataset into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # choose the model type based on whether to use SMOTE for rebalancing
    if useRebalancing:
        # configure SMOTE and under-sampling strategies
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
        # use a standard SVC model without rebalancing
        model = SVC(kernel='linear', class_weight='balanced', random_state=42)

    # train the model for the specific feature type (lbp, orb, or combined)
    model.fit(xTrain, yTrain)
    
    # save the model if requested
    if (args.save_model):
        datasetType = "complete" if args.all_emotions else "reduced"
        balancedType = "unbalanced" if args.unbalance else "balanced"
        with open(f'{datasetType}_{balancedType}_{featureType}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    # get the classification report, confusion matrix, class accuracies, and display the results
    yPred = model.predict(xTest)
    accuracyScore = accuracy_score(yTest, yPred)
    classificationReport = classification_report(yTest, yPred, zero_division=0)
    confusionMatrix = confusion_matrix(yTest, yPred)
    classAccuracies = confusionMatrix.diagonal() / confusionMatrix.sum(axis=1)
    displayResults(confusionMatrix, classAccuracies, accuracyScore, classificationReport, emotions, featureType, emotionImageCount)

# train and evaluate models
trainAndEvaluate(lbpFeatures, yLabels, 'LBP', not args.unbalance)
trainAndEvaluate(orbFeatures, yLabels, 'ORB', not args.unbalance)
trainAndEvaluate(combinedFeatures, yLabels, 'Combined', not args.unbalance)