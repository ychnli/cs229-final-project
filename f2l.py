import os
import numpy as np
from numpy import genfromtxt
import csv

FRAMES_PER_SEQUENCE = 6
OVERLAP = 3
NUM_FEATURES = 38 #ADJUST BASED ON # FEATURES IN KEYPOINTS.CSV

#Step 1: loading keypoints.csv
def loadData(keypointDir):
    dtype = [('column1', 'U10')] + [('column{}'.format(i), float) for i in range(2, NUM_FEATURES + 1)]
    temp = np.genfromtxt(keypointDir, delimiter=',', dtype=dtype)
    keypoints = np.array([list(row) for row in temp])
    return keypoints

#Step 2: generate output np array with reformatted data
def vectorize(keypoints):
    output = []
    height = np.shape(keypoints)[0]
    while(height > 1):
        sequences = []
        iterator = 0
        sequence = int(float(keypoints[0][1]))
        for row in keypoints:
            iterator += 1
            if int(float(row[1])) != sequence: break 
            new = True #assume each elem is a new one
            for i in range(len(sequences)):
                if int(float(row[3])) == int(float(sequences[i][3])):
                    new = False
                    sequences[i] = np.concatenate((sequences[i], row[4:]))
                    break
            if new: sequences.append(row)
        #out of the sequence
        for elem in sequences:
            output.append(elem)
        keypoints = keypoints[iterator-1:]
        height = np.shape(keypoints)[0]
        continue
    output = np.array(output)
    startFrame = np.array([(int(float(row[1])) * 3 - 2)for row in output])
    endFrame = startFrame + 6
    labels = np.full(np.shape(endFrame), -1)
    output = np.insert(output, 2, startFrame, axis=1)
    output = np.insert(output, 3, endFrame, axis=1)
    output = np.insert(output, 6, labels, axis=1)
    return output

#Step 3: save output into csv file
def save(output, outputDir):
    with open(outputDir, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all lines to the CSV file
        for row in output:
            writer.writerow([cell.decode('utf-8') if isinstance(cell, bytes) else str(cell) for cell in row])

rawvideos_dir = "raw"
output_dir = "labelled"
model = "yolov8l"

videos_to_process = [f"{rawvideos_dir}/{file}" for file in os.listdir(rawvideos_dir)]

for video_fp in videos_to_process:
    output_name = f"{video_fp[len(rawvideos_dir)+1:][:-4]}" #name of video w/out extension
    keypointDir = f"{output_dir}/{output_name}/keypoints.csv"
    print(keypointDir)
    outputDir = f"{output_dir}/{output_name}/unlabelledtraining.csv"

    keypoints = loadData(keypointDir)
    output = vectorize(keypoints)
    save(output, outputDir)
    
