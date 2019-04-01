# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:32:00 2018

@author: abhinav.jhanwar
"""


import face_recognition
import cv2
from sklearn.svm import SVC
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
            base = "known_people/"
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
    
  
    # Create and train the SVM classifier
    svm_clf = SVC(kernel='linear', random_state=42, probability=True)
    svm_clf.fit(known_face_encodings, known_face_names)

    return svm_clf


'''# STEP 1: Train the SVM classifier and save it to disk
print("Training SVM classifier...")
classifier = train()
print("Training complete!")
'''
pickle_in = open(base_dir+"svm_clf.pickle","rb")
classifier = pickle.load(pickle_in)
pickle_in.close()
        
base_data = pd.DataFrame()
confidence_threshold = 0.4
base_validate = "testing"
#base_test = "tester"
predictions = list()
dirs = list()
confidences_pred = list()

for img_path in tqdm(glob.glob(os.path.join(base_validate, "*.jpg"))):
    image_data = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(image_data, 1 / 255, (416, 416),
                     [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
    (frame_height, frame_width) = image_data.shape[:2]
    confidences = []
    boxes = []
    face_locations = []
    for out in outs:
        #print(out)
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
        face_locations.append(np.array([top, left+width, top+height, left
                     ]))
    
    if len(face_locations)!=1:
        print(len(face_locations), img_path, "is not suitable")
        dirs.append(img_path)
        predictions.append("Not Defined")
        confidences_pred.append(0)
        continue
    
    face_encodings = face_recognition.face_encodings(image_data, face_locations, num_jitters=10)
    probabilities = classifier.predict_proba(face_encodings)[0]
    confidence = max(probabilities)
    confidences_pred.append(confidence)
    
    if confidence>confidence_threshold:
        name = str(classifier.classes_[np.argmax(probabilities)])
        predictions.append(name)
    else:
        name = "Unknown"
        predictions.append(name)
    dirs.append(img_path)
    print("\nProbability:", probabilities, "\nName:", name, "\nConfidence:", confidence)
'''
base_data['predicted'] = predictions
base_data['dir'] = dirs  
base_data['confidence'] = confidences_pred  
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
print("Average Confidence:", sum([confidence for confidence in confidences_pred if confidence>0.7])/len([confidence for confidence in confidences_pred if confidence>0.7]))
# Rachel Green, Chandler Bing, Elon Musk'''