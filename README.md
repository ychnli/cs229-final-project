# 3 Punch Man: CV-powered Boxing Punch Classificaton
Authors: Yuchen Li, Kim Ngo, Saahil Sundaresan, Ethan Zhang

## Overview
Boxing is a relatively simple sport. Fighters utilize a set of merely 6 punches: hook, cross, right/left jab, and right/left hook. Automatic classification of a fighter's punches from match footage has high use potential for boxers: being able to quantitatively understand one's own tendencies or the tendencies of a future opponent can be invaluable, and automating this process streamlines tactical preparation. This project aims to create a machine learning model capable of taking in YOLOv8-detected keypoints on match footage and predicting whether or not sequences of frames contain punches.

## Model Training Pipeline:
1. Download .mp4 files of boxing footage into /raw.
2. Run _preprocess-script.py_ to generate keypoints.csv, /labelled/{videoname}/frames, and /labelled/{videoname}/sequences. /labelled/{videoname}/sequences contains the video broken down into 6-frame sequences with each person's ID drawn over their bounding box for ease of labeling.
3. Run _feature-engineering.py /labelled_ to generate features.csv.
4. Run _f2l.py_ to generate unlabelled.csv. Label the 7th column of the spreadsheet: 0 = no punch, 1 = straight punch (jab or cross), 2 = hook, 3 = uppercut.

From there, upload labeled csv files to Google Drive folder and run any model.ipynb on Colab. 

## Visualizing Predictions:
Using any model for predictions, replace column 7 in an unlabelled.csv spreadsheet with predicted punches. Run visualizer.py.

## Models:
BiLSTM + Softmax:
[final model](https://drive.google.com/file/d/1U4C6DinQhtv6DN-3eMsagfOpF3EpTy3t/view?usp=sharing)

BiLSTM + SVM:
[lstm feature extraction](https://drive.google.com/file/d/1vTh2idgYYzbAJSZoQwx-QqfdkjpcZttd/view?usp=sharing), 
[svm model](https://drive.google.com/file/d/1uJktywVXUeWumO0I60KBRexnmyS9Ce92/view?usp=sharing)
