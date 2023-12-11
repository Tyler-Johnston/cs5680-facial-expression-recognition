# Facial Expression Recognition Project

## Overview
This project is based on the paper "Facial Expression Recognition with LBP and ORB Features" by Ben Niu, Zhenxing Gao, and Bingbing Guo. It focuses on training models for facial expression recognition and predicting emotions in images using LBP, ORB, and their combined feature sets using an SVM.

## Installation and Execution
1. Install dependencies: `pip install -r requirements.txt`.
2. To train the model with various parameters (sampling techniques and datasets), use `python classification.py`. Include flags as needed:
   - `--save_model`: To save the trained model.
   - `--unbalance`: To run without any sampling techniques.
   - `--all_emotions`: To include all emotions in the dataset.
3. To predict emotions in a specific image using a trained model, run: `python main.py [PathToImage] [PathToModel]`.