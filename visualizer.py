import numpy as np
import cv2
import os

FRAMES_PER_SEQUENCE = 6
OVERLAP = 3
num_features = 595


vidName = input("Enter the name of the video. (Make sure the vid name is the same as its directory's name!)")
frames_fp = f"labelled/{vidName}/frames"
boxes_fp = f"labelled/{vidName}/boxes.csv"
frameDir = f"labelled/{vidName}/frames"
predictionDir = f"labelled/{vidName}/predictions"
if not os.path.exists(predictionDir):
    os.makedirs(predictionDir)

dtype = [('column1', 'U10')] + [('column{}'.format(i), int) for i in range(2, 8)] + [('column{}'.format(i), float) for i in range(8, num_features + 1)]
temp = np.genfromtxt(f"labelled/{vidName}/predictions.csv", delimiter=',', dtype=dtype)
predictions = np.array([list(row) for row in temp])

def numToPunch(num):
    num = int(num)
    if num == 0:
        return ""
    if num == 1:
        return "Jab"
    if num == 2:
        return "Hook"
    if num == 3:
        return "Uppercut"
    return "you done fucked up"

def label_person_ids(boxes_path):
    data = np.genfromtxt(boxes_path, delimiter=',')[:,1:] #boxes
    n_sequences = int(data[-1,0])

    for seq in range(1, n_sequences+1):  
        if seq % 100 == 0:
            print(f"Sequence {seq}") 
        sequence_data = data[data[:,0]==seq,:] #m x 7 array. m = 6 x n (# of people present in all 6 frames)
        p_seq_data = predictions[predictions[:,1]==str(seq),:] #n x 595 array VERY SCUFFED BUT I DONT WANT TO FIX IT
        if len(sequence_data) == 0: continue # no drawing numbers on sequences w/ no people detected
        # if len(p_seq_data) == 0: continue #SAFEGUARD FOR MANUAL LABELING TABLE ROW REMOVALS. CAN COMMENT OUT IF BEING USED FOR PREDICTIONS
        beg_frame = int(sequence_data[0,1])
        #print(f"SEQ: {seq} BEG FRAME: {beg_frame}")
        end_frame = int(sequence_data[-1,1])
        for frame_num in range(beg_frame, end_frame+1):
            # find the frame image to edit
            frame_name = f'labelled/{vidName}/frames/frame_{frame_num:04d}' #frame_xxxx
            frame_img = cv2.imread(f'{frame_name}.jpg')

            # label the person ID on the bounding box
            frame_data = sequence_data[sequence_data[:,1]==frame_num,:] # n x 7 array
            
            for i in range(frame_data.shape[0]): # for each person
                x1,y1 = frame_data[i,3],frame_data[i,4]
                person_ID = int(frame_data[i,2])
                #print(f"SEQ: {seq} PSEQ DATA: {p_seq_data} PERSON = {i} WHERE = {np.where(p_seq_data[:,5] == str(person_ID))}")
                index = np.where(p_seq_data[:,5] == str(person_ID))[0][0] #scuffed again
                punchType = numToPunch(p_seq_data[index, 6])

                cv2.putText(frame_img, punchType, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            cv2.imwrite(f'{predictionDir}/frame_{frame_num:04d}_pred.jpg', frame_img)


def makeMovie():
    # Get the list of image files in the directory
    temp = [f for f in os.listdir(predictionDir)]
    images = sorted(temp)
    print(images)

    # Open the first image to get the dimensions
    first_image = cv2.imread(os.path.join(predictionDir, images[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can try 'XVID' if 'mp4v' doesn't work
    video_writer = cv2.VideoWriter(f"labelled/{vidName}/prediction.mp4", fourcc, 15, (width, height))

    # Iterate over each image and write it to the video file
    numImages = len(images)
    for i in range(numImages):
        if i % 250 == 0:
            print(f"On image {i} out of {numImages}!")
        image_path = os.path.join(predictionDir, images[i])
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

    print("Video created successfully!")


#Run
label_person_ids(boxes_fp)
makeMovie()