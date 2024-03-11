import sys
import numpy as np
import pandas as pd
import math
import os
"""
Run on single file:
python feature_engineering.py path/to/keypoints.csv

Run on labelled directory:
python feature_engineering.py path/to/labelled
"""
def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_angle(p1, p2, p3):
    # Calculate vectors
    ux = p1[0] - p2[0]
    uy = p1[1] - p2[1]
    vx = p3[0] - p2[0]
    vy = p3[1] - p2[1]

    # Calculate dot product
    dot_product = ux * vx + uy * vy

    # Calculate magnitudes
    magnitude_u = math.sqrt(ux**2 + uy**2)
    magnitude_v = math.sqrt(vx**2 + vy**2)

    # Calculate angle in radians
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))

    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

# Features for frame
def get_features(data):
  num_examples = data.shape[0]
  features = np.zeros((num_examples, 64))

  for i in range(num_examples):

    f = 0

    # 1. Get distances between keypoints 5-12
    #nondet points set to -1
    for j in range(10, 24, 2):
      for k in range(j + 2, 25, 2):
        p1 = (data[i][j], data[i][j + 1])
        p2 = (data[i][k], data[i][k + 1])
      
        if p1 != (0, 0) or p2 != (0, 0):
          features[i][f] = calculate_distance(p1, p2)
          features[i][f + 1] = 1
        
        f += 2

    # 2. Get angles for left side
    #nondet points set to -180 degrees
    # between wrist, elbow, and shoulder
    kp5 = (data[i][10], data[i][11])
    kp7 = (data[i][14], data[i][15])
    kp9 = (data[i][18], data[i][19])
    if (0, 0) not in [kp5, kp7, kp9]:
      features[i][f] = calculate_angle(kp5, kp7, kp9)
      features[i][f + 1] = 1
    f += 2

    # between elbow, shoulder, and hip
    kp11 = (data[i][22], data[i][23])
    if (0, 0) not in [kp5, kp7, kp11]:
      features[i][f] = calculate_angle(kp7, kp5, kp11)
      features[i][f + 1] = 1
    
    f += 2

    # 3. Get angles for right side
    #nondet points set to -180 degrees
    # between wrist, elbow, and shoulder
    kp6 = (data[i][12], data[i][13])
    kp8 = (data[i][16], data[i][17])
    kp10 = (data[i][20], data[i][21])

    if (0,0) not in [kp6, kp8, kp10]:
      features[i][f] = calculate_angle(kp6, kp8, kp10)
      features[i][f + 1] = 1
    f += 2

    # between elbow, shoulder, and hip
    kp12 = (data[i][24], data[i][25])

    if (0,0) not in [kp8, kp6, kp12]:
      features[i][f] = calculate_angle(kp8, kp6, kp12)
      features[i][f + 1] = 1
    f += 2

  return features

""" Fills in missing distance and angle features 
with the mean value of the respective feature"""
def fill_in_missing_vals(features):
    feature_num = features.shape[1]
    num = 0
    count = 0
    f = 0
    while f < feature_num:
        total = 0
        count = 0
        for i in range(len(features)):
            if features[i][f + 1] == 1:
                count += 1
                total += features[i][f]
        mean = total / count
        for i in range(len(features)):
            if features[i][f + 1] == 0:
                features[i][f] = mean
        f += 2
    return features

def main(fp):
    keypoints_data = pd.read_csv(fp, header=None)
    keypoints_arr = keypoints_data.loc[:, 4 :].to_numpy(dtype=float)

    features = get_features(keypoints_arr)
    features = fill_in_missing_vals(features)
    features_df = pd.DataFrame(features)
    feature_data = pd.concat([keypoints_data, features_df], axis=1)

    features_fp = "/".join(fp.split("/")[:-1]) + "/" + "features.csv"
    feature_data.to_csv(features_fp)

if __name__ == '__main__':
    n = len(sys.argv)

    if n != 2:
        print("Usage: python script.py directory_path")
        sys.exit(1)

    directory_path = sys.argv[1]

    # Check if the provided path is a directory
    if not os.path.isdir(directory_path):
        main(sys.argv[2])

    # Loop through subdirectories
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        # Check if the current item is a directory
        if os.path.isdir(subdir_path):
            # Look for the keypoints.csv file in the current subdirectory
            keypoints_file = os.path.join(subdir_path, 'keypoints.csv')
            if os.path.isfile(keypoints_file):
                # Call the main() function with the keypoints.csv file
                main(keypoints_file)