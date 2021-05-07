# ICH_detection
NYU Deep Learning Spring 2021 Final Project
Brian Murphy and Nivedha Satyamoorthi

## Data
Download Kaggle data to the ICH_detection/data subdirectory from: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data

All data used for this project will be in the stage_2_train subdirectory
All labels used for this project are in stage_2_train.csv

## DenseNet121
To train the DenseNet121 model on the RSNA ICH training images, run ICH_detection/DenseNet/train_model.py after setting up a python/conda environment with necessary packages
Once training is finished, run ICH_detection/DenseNet/evaluate.py to get test results
To explore hyperparameters on a smaller data subset, run ICH_detection/DenseNet/train_model_sub.py

## Resnet18
