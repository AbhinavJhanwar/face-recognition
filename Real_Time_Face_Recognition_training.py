# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:25:10 2018

@author: abhinav.jhanwar
"""

# face recognition modules
import face_recognition
import cv2
# classifier modules
from sklearn.svm import SVC
# model evaluation modules
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# mathematical tools
import numpy as np
from tqdm import tqdm
# directory modules
import os
import glob
# models saving modules
import pickle
# json file handling module
import json

# read configuration file
with open('config_training.json', 'r') as outfile:  
    config = json.load(outfile)
base_dir = config['base_dir']
model_cfg = config['model_cfg']
model_weights = config['model_weights']
user_images = config['ip_user_images']

# define deep neural network parameters
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

###########################################################################
############################ training module ##############################
###########################################################################
def train(verbose=True, names=[''], known_face_encodings=[], known_face_names=[]):
    if len(names[0])>0:
        # looping through names to be trained
        for name in tqdm(names):
            # load images of person to be trained
            base = user_images
            name = name.strip()
            base = os.path.join(base, name)
            # looping through images of person to be trained
            for img_path in tqdm(glob.glob(os.path.join(base, "*.jpg"))):
                # read image
                image_data = cv2.imread(img_path)
                # load model parameters
                blob = cv2.dnn.blobFromImage(image_data, 1 / 255, (416, 416),
                                 [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                # fetch predictions from model/network
                layers_names = net.getLayerNames()
                outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
                # fetch size of image
                (frame_height, frame_width) = image_data.shape[:2]
                # declare overall confidence list
                confidences = []
                # declare bounding boxes list
                boxes = []
                # looping through model predictions/ predictions for each grid cell
                for out in outs:
                    # looping through detectors outputs for grid cell
                    for detection in out:
                        # fetch classifier probabilities for different classes
                        scores = detection[5:]
                        # fetch maximum probabilty class
                        class_id = np.argmax(scores)
                        # define confidence as maximum probability
                        confidence = scores[class_id]
                        # filter predictions based on confidence threshold
                        if confidence > conf_threshold:
                            # fetch bounding box dimensions
                            center_x = int(detection[0] * frame_width)
                            center_y = int(detection[1] * frame_height)
                            width = int(detection[2] * frame_width)
                            height = int(detection[3] * frame_height)
                            left = int(center_x - width / 2)
                            top = int(center_y - height / 2)
                            # append confidence in confidences list
                            confidences.append(float(confidence))
                            # append bounding box in bounding boxes list
                            boxes.append([left, top, width, height])
                
                # perform non maximum suppression of overlapping images
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                           nms_threshold)
                
                # fetch faces bounding boxes
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
                    
        
        # save the encodings
        pickle_out = open(base_dir+"known_face_names.pickle","wb")
        pickle.dump(known_face_names, pickle_out)
        pickle_out.close()
        pickle_out = open(base_dir+"known_face_encodings.pickle","wb")
        pickle.dump(known_face_encodings, pickle_out)
        pickle_out.close()
        
        # Create and train the SVM classifier
        svm_clf = SVC(C=100, gamma=0.0001, kernel='rbf', probability=True, verbose=True)
        grid = svm_clf
        
        # GridSearch for parameters optimization
        '''param_grid = {
                    'C': [100, 50, 10, 5, 1, 0.5, 0.1],
                    'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                    'kernel' :['rbf', 'linear']
                    }
        grid = GridSearchCV(svm_clf, param_grid, verbose=3, n_jobs=2)'''
        grid.fit(known_face_encodings, known_face_names)
        
        #print("Parameters selected for model:", grid.best_estimator_)
        
        # save classifier
        pickle_out = open(base_dir+"svm_clf.pickle","wb")
        pickle.dump(grid, pickle_out)
        pickle_out.close()
    else:
        print("Encoding Skipped!\n")

if __name__ == '__main__':
    # training
    nms_threshold = 0.3
    conf_threshold = 0.7
    names=[]
    for user in glob.glob(os.path.join(user_images,'*')):
        names.append(user.split('\\')[-1])
        
    if names!=[]:
        print("Training... ", names)
        known_face_encodings=[]
        known_face_names=[]
        # start training for names provided in user input
        train(verbose=True, names=names, known_face_encodings=known_face_encodings, known_face_names=known_face_names)
            
    else:
        print("No folders detected. Exiting...")