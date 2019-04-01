# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:34:45 2018

@author: abhinav.jhanwar
"""

''' hog + k-Neighbor'''

import face_recognition
import cv2
from imutils.video import FPS
import pickle
import numpy as np
import pandas as pd
from sklearn import neighbors
import os, glob
import math
from collections import defaultdict

base_dir = "SavedModels/"

# load pre-traind encodings
pickle_in = open(base_dir+"known_face_encodings.pickle","rb")
known_face_encodings = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(base_dir+"known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)
pickle_in.close()

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # get name of person to be trained
    names = input("Write names of persons to be trained:\n").split(',')
    
    if len(names[0])>0:
        print("Encoding Images...\n")
        for name in names:
            base = "known_people"
            name = name.strip()
            # find images of person to be trained
            base = os.path.join(base, name)
            for img_path in glob.glob(os.path.join(base, "*")):
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image, model="cnn")
                if len(face_locations) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    known_face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=5)[0])
                    known_face_names.append(name)
                    
        # save the trained models
        pickle_out = open(base_dir+"known_face_names.pickle","wb")
        pickle.dump(known_face_names, pickle_out)
        pickle_out.close()
        
        pickle_out = open(base_dir+"known_face_encodings.pickle","wb")
        pickle.dump(known_face_encodings, pickle_out)
        pickle_out.close()
        
        print("Encoding Completed!\n")
    
    else:
        print("Encoding Skipped!\n")
    
  
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(known_face_names))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance', n_jobs=-1)
    knn_clf.fit(known_face_encodings, known_face_names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


# STEP 1: Train the KNN classifier and save it to disk
print("Training KNN classifier...")
classifier = train("knn_examples/train", 
                   model_save_path="knn_model.clf",
                   knn_algo='auto',
                   n_neighbors=2)
print("Training complete!")


#classifier = pickle.load("knn_model.clf")

distance_threshold=0.4

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

fps = FPS().start()
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_encoding = [face_encoding]
        closest_distances = classifier.kneighbors(face_encoding, n_neighbors=5)
        # create name: distance dict
        dist_name = defaultdict(list)
        [dist_name[known_face_names[key]].append(value) for value, key in zip(closest_distances[0][0], closest_distances[1][0])]
        # get max frequent name in dict
        max_count=0
        for key in dist_name.keys():
            count = len(dist_name[key])
            if count>max_count:
                max_count=count
                avg_name = key
            
        avg_distance = round(sum(dist_name[avg_name])/len(dist_name[avg_name]),2)
        
        name = "Unknown"
        #print(closest_distances)

        # If a match was found in known_face_encodings, just use the first one.
        if avg_distance<= distance_threshold:
            name = avg_name
            distance = avg_distance
            #print(name, "Distance: ", face_distances[first_match_index])

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
        
        if name!='Unknown':
            # label for confidence
            cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, str(distance), (left + 6, top-3), font, 0.5, (255, 255, 255), 1)


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