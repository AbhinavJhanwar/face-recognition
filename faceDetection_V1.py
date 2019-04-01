# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:22:57 2019

@author: abhinav.jhanwar
"""

''' yolo detection,
    encoding dlib,
    knn classification'''
    
import face_recognition
import cv2
# mathematical tools
import numpy as np
# time modules
from time import time
# models saving modules
import _pickle as pickle
import imutils
from collections import defaultdict

class faceDetection:
    __slots__ = 'base_dir', 'nms_threshold', 'yolo_conf_threshold',\
    'classifier_threshold', 'knn_distance_threshold', 'net', 'known_face_names',\
    'model','resnet_conf_threshold'
    
    def __init__(self, config):
        self.model = config['faceDetector']
        if self.model == 'yolo':
            model_cfg = config['yolo_model_cfg']
            model_weights = config['yolo_model_weights']
            self.nms_threshold = config['nms_threshold']
            self.yolo_conf_threshold = config['yolo_conf_threshold']
            
            # define deep neural network parameters
            self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        elif self.model == 'resnet':
            modelFile = config['resnet_modelFile']
            configFile = config['resnet_model_cfg']
            self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
            self.resnet_conf_threshold = config['resnet_conf_threshold']
        
        self.base_dir = config['base_dir']
        self.classifier_threshold = config['classifier_threshold']
        self.knn_distance_threshold = config['knn_distance_threshold']
            
        pickle_in = open(self.base_dir+"known_face_names.pickle","rb")
        self.known_face_names = pickle.load(pickle_in)
        pickle_in.close()
        
    def detectFace(self, frame):
        
        ##################################
        # apply image processing
        #######################################
        #frame = imutils.resize(frame, height=200)
        
        if self.model == 'yolo':
            # load model parameters
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416),
                         [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            
            # fetch model predictions
            layers_names = self.net.getLayerNames()
            outs = self.net.forward([layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()])
            
            # fetch captured image dimensions
            (frame_height, frame_width) = frame.shape[:2]
            
            # declare confidences, bounding boxes and face location bounding boxes list
            confidences = []
            boxes = []
            face_locations = []
                
            # looping through grid cells
            for out in outs:
                # looping through detectors
                for detection in out:
                    # fetch classes probability
                    scores = detection[5:]
                    # fetch class with maximum probability
                    class_id = np.argmax(scores)
                    # fetch maximum probability
                    confidence = scores[class_id]
                    # filter prediction based on threshold value
                    if confidence > self.yolo_conf_threshold:
                        # fetch validated bounding boxes
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        width = int(detection[2] * frame_width)
                        height = int(detection[3] * frame_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        # add confidences and bounding boxes in list
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
            
            # perform non maximum suppression to remove overlapping images based on nms_threshold value           
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_conf_threshold,
                                       self.nms_threshold)
            
            # fetch legitimate face bounding boxes
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                face_locations.append(np.array([max(0, top), min(left+width, frame_width), min(top+height, frame_height), max(left, 0)
                             ]))
        
        elif self.model == 'resnet':
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # fetch captured image dimensions
            (frame_height, frame_width) = frame.shape[:2]

            # define face location list
            face_locations = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > self.resnet_conf_threshold:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    location = tuple(box.astype("int"))
                    face_locations.append((location[1], location[2], location[3], location[0]))
                   
        return face_locations
    
    def getFaceID(self, frame, face_locations):
        
        faces = []
        
        # load classifier
        pickle_in = open(self.base_dir+"knn_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        # encode faces to be fed to classifier for prediction
        face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=1)
        
        # loop through face encodings and face boundary boxes
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_encoding = [face_encoding]
            # fetch probability distribution of predicted classes
            predictions = classifier.kneighbors(face_encoding, n_neighbors=100)
     
            dist_name = defaultdict(list)
            [dist_name[self.known_face_names[key]].append(value) for value, key in zip(predictions[0][0], predictions[1][0])]
            
            # sort dictionary based on number of values
            dist_name = sorted(dist_name.items(), key=lambda item: len(item[1]), reverse=True)
           
            # fetch average distance of top class from given image
            avg_distance = round(sum(dist_name[0][1])/len(dist_name[0][1]),2)
            confidence = (len(dist_name[0][1])/100)
            
            name = "Unknown"
            if (avg_distance <= self.knn_distance_threshold and confidence>=self.classifier_threshold) or (avg_distance <= self.knn_distance_threshold+0.06 and confidence>=0.9):
                name = dist_name[0][0]
            
            faces.append((name, confidence))
            
            
            print("[INFO] Average Distance:", avg_distance)
            print("[INFO] Probability:", confidence)
            print("[INFO] Class:", dist_name[0][0], "\n")
            
            '''try:
                print("########################################################")
                for i in range(len(dist_name)):
                    print("[INFO] Average Distance:", round(sum(dist_name[i][1])/len(dist_name[i][1]),2))
                    print("[INFO] Probability:", (len(dist_name[i][1])/100))
                    print("[INFO] Class:", dist_name[i][0], "\n")
                print("########################################################")
            
            except:
                pass'''
        return faces

    