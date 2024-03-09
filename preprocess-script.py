import subprocess
import sys

def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully!")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}.")

# Install the ultralytics package if not already installed
install_package("ultralytics")

import numpy as np
from ultralytics import YOLO
import cv2
import os
import csv
import shutil

FRAMES_PER_SEQUENCE = 6
OVERLAP = 3
MAX_DETECTIONS = 5

#Step 1: Extract keypoints from video
def extractKeypoints(video_fp,outputsDirectory,model):
    """
    Extract and return keypoints tensor from video_fp using YOLO 2D pose estimation model
    """

    model = YOLO(f"{model}-pose.pt")
    results = model(video_fp, project=output_dir, name=output_name, stream=False, save=True,max_det=5, save_conf=True, vid_stride=2, conf=0.4)
    return results

#Step 2: split labelled keypoints video into a folder of individual frames at dir frames_fp
def splitVideoToFrames(labelledvideo_fp,frames_fp):
    """
    Split video of labelled keypoints at labelledvideo_fp into individual frames
    Save in directory frames_fp as jpg files
    """
    if not os.path.exists(frames_fp): 
        os.makedirs(frames_fp)
    video = cv2.VideoCapture(labelledvideo_fp)
    if not video.isOpened():
        print(f"Error opening video {labelledvideo_fp}")
    i = 1
    while True:
        ret, frame = video.read()
        if not ret: break
        frame_filename = os.path.join(frames_fp, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        i += 1
    video.release()

#Step 3: Assign person IDs to each character by sequence

def assignIDNumbers(framesDir, centroid_thresh=0.15, cossim_weight = 1, bb_weight = 1.75):
    """
    Assign person IDs to each character in each sequence
    Ordering:

    Return value: dictionary of [1-indexed sequence number]-> dictionary of
    [1-indexed character number]->[index in keypoints for each frame in sequence]

    Assignment order:
     - If a person doesn't appear in the first frame of a sequence don't classify him.
     - One-indexed
    """

    def modified_cossim(v1, v2):
        """
        Helper function
        Modified cosine similarity: only nonzero elements are considered
        """
        v1p = np.where((v1 != 0) & (v2 != 0), v1, 0)
        v2p = np.where((v1 != 0) & (v2 != 0), v2, 0)
        if np.sum(v1p)==0 or np.sum(v2p)==0:
            return 0
        return (v1p @ v2p)/(np.linalg.norm(v1p)*np.linalg.norm(v2p))

    def sortfn(x):
        """
        Helper function
        We will order people left-to-right by the minimum x coordinate of their nonzero
        keypoints in the first frame of the sequence
        """
        arr = np.array(sequence[0].keypoints.xy[x-1].tolist())[:,0]
        if len(arr[arr>0])>0:
            return arr[arr > 0].min()
        return 0

    def calc_centroid(bounding_box):
        """
        Helper function: calculate the centroid of the bounding box (an xyxy or xyxyn array)
        """
        return np.array([(bounding_box[0]+bounding_box[2])/2, 
                         (bounding_box[1]+bounding_box[3])/2])
    
    def calc_size(bounding_box):
        """
        Helper function: calculate the size of a bounding box
        """
        return np.abs(bounding_box[2] - bounding_box[0]) * np.abs(bounding_box[3] - bounding_box[1])
        
    def findCameraChanges(framesDir, threshold=0.9):
        """
        Helper function
        Detect camera angle changes in a directory of frames. Return a list of zero-indexed frame numbers of such changes
        """
        angle_change_frames = []
        prev_hist = None

        files = sorted(os.listdir(framesDir))

        for file in files:
            if not file.endswith('.jpg'): continue  # Skip non-JPG files

            frame_path = os.path.join(framesDir, file)
            frame = cv2.imread(frame_path)

            if frame is None: continue #skip unreadable frames

            curr_hist = cv2.calcHist([frame], [0], None, [256], [0, 256])

            if prev_hist is not None:
                # Compare curr frame's histogram with prev frame's histogram
                similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

                if similarity < threshold:
                    angle_change_frames.append(int(file[-8:-4])-1)

            prev_hist = curr_hist

        return angle_change_frames

    ID_assignments = dict()
    num_frames = len(results)

    angleChangeFrames = findCameraChanges(framesDir)
    
    # here the iterator s is the true (external) frame of video 
    for s in range(0,num_frames,OVERLAP):
        #print(f"Processing sequence {int(s/OVERLAP) + 1}")
        sequence = results[s:s+FRAMES_PER_SEQUENCE]
        if len(sequence) != FRAMES_PER_SEQUENCE: break #means we're done processing

        #ensure that the sequence doesn't contain any frames with camera angle changes
        skip = False
        for i in range(s,s+FRAMES_PER_SEQUENCE):
            if i in angleChangeFrames:
                skip = True
                break
        if skip: continue

        #ensure that the sequence doesn't contain any frames without ID'd characters
        skip = False
        for i in range(FRAMES_PER_SEQUENCE):
            if len(sequence[i].keypoints.xy[0])==0:
                skip = True
                break
        if skip: continue
        
        # assign ID labels for the first frame 
        num_chars_first_frame = len(sequence[0].keypoints.xy)
        characters = np.full((MAX_DETECTIONS, FRAMES_PER_SEQUENCE), -1, dtype=int)
        for i in range(num_chars_first_frame):
            characters[i, 0] = i
                
        # match indexes at curr internal frame to those from prev internal frame
        for internal_frame in range(1, FRAMES_PER_SEQUENCE):
            positional_indexes = [r for r in range(len(sequence[internal_frame].keypoints.xy))]
            
            curr_keypoints = np.array(sequence[internal_frame].keypoints.xy) # shape: (n_people, 17, 2)
            curr_boxes = np.array(sequence[internal_frame].boxes.xyxyn) # shape: (n_people, 4) 
            curr_n_people = curr_keypoints.shape[0]
            
            # calculate sizes of boxes
            curr_box_sizes = np.zeros((curr_n_people, 2))
            for curr_person in range(curr_n_people):
                curr_box_sizes[curr_person] = [curr_person, calc_size(curr_boxes[curr_person])]
            sorted_sizes = curr_box_sizes[np.argsort(-curr_box_sizes[:, 1])]
            #print(sorted_sizes)
            
            prev_keypoints = np.array(sequence[internal_frame-1].keypoints.xy)
            prev_boxes = np.array(sequence[internal_frame-1].boxes.xyxyn) 
            prev_n_people = prev_keypoints.shape[0]
                        
            # for each person in the current frame with sufficiently big bounding box 
            # calculate their score with each person in prev frame
            scores = []
            for curr_person in sorted_sizes[0:3,0]:   
                curr_person = int(curr_person)
                for prev_person in range(prev_n_people):
                    curr_centroid = calc_centroid(curr_boxes[curr_person])
                    prev_centroid = calc_centroid(prev_boxes[prev_person])
                    
                    cossim = modified_cossim(curr_keypoints[curr_person].flatten(),
                                            prev_keypoints[prev_person].flatten())
                    centroid_dist = np.linalg.norm(curr_centroid - prev_centroid)
                    
                    # if cossim is negative, or centroid_dist is too far, don't even consider the pair
                    if cossim < 0 or centroid_dist > centroid_thresh:
                        continue 
                    
                    # calculate the similarity score between each pair of people 
                    similarity_score = cossim_weight*cossim - bb_weight*centroid_dist
                    scores.append([curr_person, prev_person, similarity_score])
                
            scores = np.array(scores) 
            # sort the scores based on similarity 
            if scores.shape[0] > 1:
                sorted_scores = scores[np.argsort(-scores[:, 2])]
            
            else:
                sorted_scores = scores
            
            # assign pairs in sorted order
            for i in range(sorted_scores.shape[0]):
                curr_person, prev_person = tuple(sorted_scores[i,0:2])
                
                # if already curr_person's prev_person is already assigned, skip
                assignment_index = np.arange(0, characters.shape[0])[characters[:,internal_frame-1]==prev_person]
                
                if assignment_index.size == 0: 
                    # print('empty assignment')
                    continue 
                if characters[assignment_index, internal_frame] != -1:
                    #print(f'Previous {prev_person} already assigned to {characters[assignment_index, internal_frame]}')
                    continue
                if curr_person in characters[:,internal_frame]:
                    #print(f'Current {curr_person} already assigned')
                    continue

                
                characters[assignment_index, internal_frame] = curr_person 
                #print(characters)
            
        # convert characters to a dict        
        characters_purged = dict()
        for i in range(characters.shape[0]):
            if -1 not in characters[i,:]:
                characters_purged[i] = list(characters[i,:])
        
        ID_assignments[int(s/OVERLAP)+1] = characters_purged #key: 1-indexed seq number, value: character assignments dict
    return ID_assignments
    

#Step 4: Write keypoints to CSV
def write_keypoints(keypoints_fp):
    """
    Write keypoints data to CSV file with the following format
     - Column 0: video ID
     - Column 1: sequence number (1-indexed)
     - Column 2: Internal frame number (0-indexed)
     - Column 3: Person ID (1-indexed)
     - Columns 4-37: keypoints (xyxy...xy)
    """
    lines = []
    
    for sequence in IDs.keys():
        seqstart = (sequence-1)*OVERLAP
        #print(f"SEQUENCE: {sequence}")
        for internal_frame in range(FRAMES_PER_SEQUENCE): #frame within sequence
            external_frame = seqstart + internal_frame #total frame
            for person in IDs[sequence].keys():
                if -1 not in IDs[sequence][person]:
                    if len(IDs[sequence][person]) > internal_frame and IDs[sequence][person][internal_frame]!=-1:
                        index = IDs[sequence][person][internal_frame]
                        #print(f"INDEX: {index}, PERSON: {person} FRAME: {internal_frame}")
                        keypoints = np.array(results[external_frame].keypoints.xy[index].tolist()).flatten().tolist()
                        line = [output_name, sequence, internal_frame, person] + keypoints

                        lines.append(line)
    with open(keypoints_fp, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all lines to the CSV file
        writer.writerows(lines)

#Step 5: Write bounding boxes to CSV
def write_boxes(boxes_fp):
    """
    Write bounding box data to CSV with the following format
     - Column 0: video ID
     - Column 1: sequence number (1-indexed)
     - Column 2: 1-indexed external frame number, 4-digit padded as string
     - Column 3: person ID
     - Columns 4-7: bounding boxes (xyxy)
    """

    lines = []
    
    for sequence in IDs.keys():
        seqstart = (sequence-1)*OVERLAP
        #print(f"SEQUENCE: {sequence}")
        for internal_frame in range(FRAMES_PER_SEQUENCE): #frame within sequence
            external_frame = seqstart + internal_frame #total frame
            for person in IDs[sequence].keys():
                if -1 not in IDs[sequence][person]:
                    if len(IDs[sequence][person]) > internal_frame and IDs[sequence][person][internal_frame]!=-1:
                        index = IDs[sequence][person][internal_frame]
                        boxes = np.array(results[external_frame].boxes.xyxy[index].tolist()).flatten().tolist()
                        line = [output_name,sequence,f"{(external_frame+1):04d}",person] + boxes
    
                        lines.append(line)
    with open(boxes_fp, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write all lines to the CSV file
        writer.writerows(lines)

def delete_dir(fp):
    """
    Delete directory at given filepath if it exists
    """
    if os.path.exists(fp) and os.path.isdir(fp):
        shutil.rmtree(fp)

#Step 6: Add person ID numbers to frame images for labelling
def label_person_ids(boxes_path):
    data = np.genfromtxt(boxes_path, delimiter=',')[:,1:]
    video_name = np.genfromtxt(boxes_path, dtype=str, delimiter=',')[0,0]
    n_sequences = int(data[-1,0])
    
    
    # Make a directory for sequence data if not already exist
    sequence_path = f'labelled/{video_name}/sequences'
    if not os.path.exists(sequence_path):
        os.makedirs(sequence_path)
        
    for seq in range(1,n_sequences+1):   
        sequence_data = data[data[:,0]==seq,:]
        if len(sequence_data) == 0: continue # no drawing numbers on sequences w/ no people detected
        beg_frame, end_frame = int(sequence_data[0,1]), int(sequence_data[-1,1])
        for frame_num in range(beg_frame, end_frame+1):
            # find the frame image to edit
            frame_name = f'labelled/{video_name}/frames/frame_{frame_num:04d}'
            frame_img = cv2.imread(f'{frame_name}.jpg')

            # label the person ID on the bounding box
            frame_data = sequence_data[sequence_data[:,1]==frame_num,:] 
            for i in range(frame_data.shape[0]):
                x1,y1 = frame_data[i,3],frame_data[i,4]
                person_ID = int(frame_data[i,2])
                cv2.putText(frame_img, str(person_ID), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            
            cv2.imwrite(f'{sequence_path}/sequence_{seq:04d}_frame_{frame_num-beg_frame}.jpg', frame_img)


rawvideos_dir = "raw"
output_dir = "labelled"
model = "yolov8l"

"""
    Run preprocessing pipeline on a given video from its filepath
    Results are in output_dir/[video name]
"""

#Provide here a list of video fp's to process. They should be stored in rawvideos_dir (which is "raw" by default) 
videos_to_process = []

#videos_to_process = [f"{rawvideos_dir}/{file}" for file in os.listdir(rawvideos_dir)]

for video_fp in videos_to_process:
    output_name = f"{video_fp[len(rawvideos_dir)+1:][:-4]}"
    #labelledvideo_fp = f"{output_dir}/{output_name}/{video_fp[len(rawvideos_dir)+1:]}"
    #AVI FOR ETHAN, use above line if you're a plebian and don't use windows.
    labelledvideo_fp = f"{output_dir}/{output_name}/{output_name}.avi"
    frames_fp = f"{output_dir}/{output_name}/frames"
    keypoints_fp = f"{output_dir}/{output_name}/keypoints.csv"
    boxes_fp = f"{output_dir}/{output_name}/boxes.csv"

    delete_dir(f"{output_dir}/{output_name}")
    
    results = extractKeypoints(video_fp,output_dir,model)
    splitVideoToFrames(labelledvideo_fp,frames_fp)
    IDs = assignIDNumbers(frames_fp)
    write_keypoints(keypoints_fp)
    write_boxes(boxes_fp)   
    label_person_ids(boxes_fp)
