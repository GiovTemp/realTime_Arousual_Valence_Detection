# Introduction
Our project focuses on applying computer vision to facial emotion recognition, with a specific emphasis on utilizing action units to predict emotions and also the intensity values of the recognized. Common practice in this context involves training models using extensive datasets or employing pre-trained models on large datasets. In our case, the approach is centered around facial recognition through action units, aiming to predict the intensity values associated with the identified emotions in images.

## main.py
The main file containing the comlete code with the real time prediction of the arousal and valence.

## data/au_pred_model.h5
The file containing the model trained to predict the facial action unit.

## data/arousal_valence_pred_model.h5
The file containing the model trained to predict the emotion arousal and valence.

## data/afew_preprocessing/preprocessing.py
The code used to preprocess the AFEW dataset

## data/afew_preprocessing/merge_csv.py
The code used to merge all the csv files present in this folder.
