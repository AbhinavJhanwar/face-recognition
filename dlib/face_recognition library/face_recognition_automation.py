# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:08:45 2018

@author: abhinav.jhanwar
"""


import face_recognition
import cv2
import os
import glob
import time
import pickle
from imutils.video import FPS

base_dir = "SavedModels/"


# load pre-traind encodings
pickle_in = open(base_dir+"known_face_encodings.pickle","rb")
known_face_encodings = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(base_dir+"known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)
pickle_in.close()

# get name of person to be trained
names = input("Write names of persons to be trained:\n").split(',')

if len(names[0])>0:
    print("Training Images...\n")
    for name in names:
        base = "known_people"
        name = name.strip()
        # find images of person to be trained
        base = os.path.join(base, name)
        for path in glob.glob(os.path.join(base, "*")):
            print(path)
            image = face_recognition.load_image_file(path)
            try:
                # using cnn to train models
                face_locations = face_recognition.face_locations(image, model="cnn")
                image_encoding = face_recognition.face_encodings(image, face_locations, num_jitters=5)[0]
                known_face_encodings.append(image_encoding)
                known_face_names.append(name)    
            except:
                print("No face Detected in ",path)
    # save the trained models
    pickle_out = open("known_face_names.pickle","wb")
    pickle.dump(known_face_names, pickle_out)
    pickle_out.close()
    
    pickle_out = open("known_face_encodings.pickle","wb")
    pickle.dump(known_face_encodings, pickle_out)
    pickle_out.close()
    
    print("Training Completed!\n")

else:
    print("Training Skipped!\n")

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

fps = FPS().start()
while True:
    #t1 = time.time()
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    # number_of_times_to_upsample: higher number to find smaller faces
    # model = "cnn" slower but accurate
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    # num_jitters : how many times to upsample the image
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        # uses SVM model for classification
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #print(matches)
        name = "Unknown"
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            ##########################################
            ### to be modified in future as follows:
            #   1) take help of matches and known_face_names lists to create a dictionary - {name1: count, name2: count}
            #   2) after creating the dictionary which contains the count of all names which are detected in the given 
            #      image for a particular face, we need to find out the max count name and that will be the name of face
            #      in the image
            #   3) to get confidence or distance check the max count name and find min distance belonging to that name in
            #      face_distances array
            ##########################################
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            #print(name, "Confidence: {0}%".format(round((1-min(face_distances))*100, 2)))
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
    
    #t2 = time.time()
    #print("Time Taken: ", round(t2-t1, 2))
    
    # Display the resulting image
    cv2.imshow('Face Recognition', frame)
    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # update the FPS counter
    fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()