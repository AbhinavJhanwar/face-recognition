# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:35:27 2018

@author: abhinav.jhanwar
"""

import face_recognition
import cv2
from sklearn import neighbors
from collections import defaultdict
import os, glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

base_dir = "SavedModels/"

known_face_encodings=[]
known_face_names=[]
model_cfg = base_dir+"yolov3-face.cfg"
model_weights = base_dir+"yolov3-wider_16000.weights"
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

nms_threshold = 0.3
conf_threshold = 0.7

def train(verbose=True):
    # get name of person to be trained
    names = input("Write names of persons to be trained:\n").split(',')
    
    if len(names[0])>0:
        print("Encoding Images...\n")
        for name in tqdm(names):
            base = "known_people"
            name = name.strip()
            # find images of person to be trained
            base = os.path.join(base, name)
            for img_path in tqdm(glob.glob(os.path.join(base, "*"))):
                image_data = cv2.imread(img_path)
                blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (416, 416)), 1 / 255, (416, 416),
                                 [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layers_names = net.getLayerNames()
                outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
                (frame_height, frame_width) = image_data.shape[:2]
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > conf_threshold:
                            center_x = int(detection[0] * frame_width)
                            center_y = int(detection[1] * frame_height)
                            width = int(detection[2] * frame_width)
                            height = int(detection[3] * frame_height)
                            left = int(center_x - width / 2)
                            top = int(center_y - height / 2)
                            confidences.append(float(confidence))
                            boxes.append([left, top, width, height])
                            
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                           nms_threshold)
                face_locations = []
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    left = box[0]
                    top = box[1]
                    width = box[2]
                    height = box[3]
                    face_locations.append(np.array([top, left + width, top + height, left
                                 ]))
                 
                if len(face_locations) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    known_face_encodings.append(face_recognition.face_encodings(image_data, known_face_locations=face_locations, num_jitters=1)[0])
                    known_face_names.append(name)
        print("Encoding Completed!\n")
    
    else:
        print("Encoding Skipped!\n")
    
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='auto', weights='distance', n_jobs=-1)
    knn_clf.fit(known_face_encodings, known_face_names)
    return knn_clf

'''# STEP 1: Train the KNN classifier and save it to disk
print("Training KNN classifier...")
classifier = train()
print("Training complete!")'''

pickle_in = open(base_dir+"knn_clf.pickle","rb")
classifier = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(base_dir+"known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)
pickle_in.close()

# STEP 2: Predictions
base_data = pd.DataFrame()
distance_threshold=0.5

base_validate = "testing"
#base_test = "tester"

predictions = []
dirs = list()
distances = list()

for img_path in tqdm(glob.glob(os.path.join(base_validate, "*.jpg"))):
    image_data = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (416, 416)), 1 / 255, (416, 416),
                     [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
    (frame_height, frame_width) = image_data.shape[:2]
    confidences = []
    boxes = []
    face_locations = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        face_locations.append(np.array([top, left + width, top + height, left
                     ]))
    
    if len(face_locations)!=1:
        print(len(face_locations), img_path, "is not suitable")
        dirs.append(img_path)
        predictions.append({"Name":"Not Defined",  "Probability":None, "Dist":None})
        continue
    
    face_encodings = face_recognition.face_encodings(image_data, face_locations)
    closest_distances = classifier.kneighbors(face_encodings, n_neighbors=10)
     
    dist_name = defaultdict(list)
    [dist_name[known_face_names[key]].append(value) for value, key in zip(closest_distances[0][0], closest_distances[1][0])]
    
    # sort dictionary based on number of values
    dist_name = sorted(dist_name.items(), key=lambda item: len(item[1]), reverse=True)
   
    # fetch average distance of top class from given image
    avg_distance = round(sum(dist_name[0][1])/len(dist_name[0][1]),2)
    if avg_distance <= distance_threshold:
        name = dist_name[0][0]
        probability = (len(dist_name[0][1])/classifier.n_neighbors)*100
    else:
        name = "Unknown"
        probability = 0
        
    predictions.append({"Name":name,  "Probability":probability, "Dist":avg_distance})
    
    
    dirs.append(img_path)
    print("\nDistances:", closest_distances, "\nName:", name)
    
'''base_data['predicted'] = predictions
base_data['dir'] = dirs  
base_data['distance'] = distances  
base_data['test'] = [
                     "Unknown", 
                     "Chandler Bing", 
                     "Rachel Green", 
                     "Chandler Bing",
                     "Rachel Green",
                     "Rachel Green",
                     "Chandler Bing",
                     "Elon Musk",
                     "Elon Musk",
                     "Elon Musk"]
base_data['val'] = [
                     "Chandler Bing", 
                     "Unknown", 
                     "Elon Musk",
                     "Unknown",
                     "Elon Musk",
                     "Rachel Green", 
                     "Rachel Green",
                     "Rachel Green",
                     "Chandler Bing",
                     "Chandler Bing"
                     ]
print("Average Distance:", sum([distance for distance in distances if distance<0.5])/len([distance for distance in distances if distance<0.5]))
# Rachel Green, Chandler Bing, Elon Musk'''