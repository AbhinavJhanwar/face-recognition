# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:43:18 2019

@author: abhinav.jhanwar
"""


''' margin applied in yolo 
    front and side face probability
    histogram equalization
    distance threshold specific to class
    using dlib'''

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
import json
import os
from tqdm import tqdm

#############################################################
import face_recognition_models
from PIL import ImageFile
import dlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])
##################################################################
    
class faceDetection:
    __slots__ = 'base_dir', 'nms_threshold', 'yolo_conf_threshold',\
    'classifier_threshold', 'knn_distance_threshold', 'net', 'known_face_names',\
    'model','resnet_conf_threshold', 'front_threshold', 'side_threshold', 'margin',\
    'knn_distance_threshold_margin'
    
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
        self.knn_distance_threshold_margin = config['knn_distance_threshold_margin']
        self.margin = config["margin_percent"]
            
        pickle_in = open(self.base_dir+"known_face_names.pickle","rb")
        self.known_face_names = pickle.load(pickle_in)
        pickle_in.close()
        
    def getDistance(self, frame):
        face_locations = self.detectFace(frame)
        faces = []
        
        # load classifier
        pickle_in = open(self.base_dir+"knn_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        ######################################################
        # histogram equalization
        for face_location in face_locations:
            #print("[INFO] Face Locations:", face_location)
            frame1 = frame[face_location[0]:face_location[2],face_location[3]:face_location[1],:]
            img_to_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
            img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
            frame1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

            frame[face_location[0]:face_location[2],face_location[3]:face_location[1],:] = frame1  
        ###################################################
        
        # encode faces to be fed to classifier for prediction
        face_encodings = self.getFaceEncoding(frame, face_locations)
          
        #cv2.imshow("image", frame)
        #cv2.waitKey(0)
        
        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
            
        # loop through face encodings and face boundary boxes
        for (top, right, bottom, left), face_encoding, face_landmark in zip(face_locations, face_encodings, face_landmarks_list):
            
            # get face pose
            face_pose = self.getFacePose(frame, top, right, bottom, left, face_landmark)
            
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
            
            faces.append((dist_name[0][0], confidence, avg_distance, face_pose))
            
            '''print("[INFO] Face Pose:", face_pose)
            print("[INFO] Average Distance:", avg_distance)
            print("[INFO] Probability:", confidence*100)
            print("[INFO] Class:", dist_name[0][0], "\n")'''
        
        return faces
        
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
                face_locations.append(np.array([max(0, top), min(left+width+(width*self.margin//100), frame_width), min(top+height, frame_height), max(left-(width*self.margin//100), 0)
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
    
    def getFacePose(self, frame, top, right, bottom, left, face_landmark):
        left_eye = ((face_landmark['left_eye'][2][0]+face_landmark['left_eye'][1][0])//2, face_landmark['left_eye'][2][1])
        right_eye = ((face_landmark['right_eye'][2][0]+face_landmark['right_eye'][1][0])//2, face_landmark['right_eye'][2][1])
        nose_tip = face_landmark['nose_tip'][2]
            
        frame_length = right-left
        # left_eye-right_eye, left_eye-nose_tip, right_eye-nose_tip
        distances = list(map(lambda x: abs(x[0][0]/frame_length-x[1][0]/frame_length), [(left_eye, right_eye), (left_eye, nose_tip), (right_eye, nose_tip)]))
        #print(distances)
        cv2.line(frame, left_eye, left_eye, (255, 255, 255), 2)
        cv2.line(frame, right_eye, right_eye, (255, 255, 255), 2)
        cv2.line(frame, nose_tip, nose_tip, (255, 255, 255), 2)
        #print('[INFO]', distances, '\n')
        font = cv2.FONT_HERSHEY_DUPLEX
        if (distances[0]<0.36 and distances[1]<0.11) or (distances[0]<0.36 and distances[2]<0.17):
            cv2.putText(frame, "SIDE POSE", (20, 40), font, 0.5, (255, 255, 255), 1)
            return 'side'
        else:
           cv2.putText(frame, 'FRONT POSE', (20, 40), font, 0.5, (255, 255, 255), 1)
           return 'front'
    
    def getFaceEncoding(self, frame, face_locations):
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]
        #pose_predictor = pose_predictor_5_point
        pose_predictor = pose_predictor_68_point
        raw_landmarks = []
        for face_location in face_locations:
            raw_landmarks.append(pose_predictor(frame, face_location))

        return [np.array(face_encoder.compute_face_descriptor(frame, raw_landmark_set, num_jitters=1)) for raw_landmark_set in raw_landmarks]
    
    def getFacialLandmarks(self, frame, face_locations):
        return face_recognition.face_landmarks(frame, face_locations)
        
    def getFaceID(self, frame, face_locations):
        
        ######################################################
        # histogram equalization
        for face_location in face_locations:
            #print("[INFO] Face Locations:", face_location)
            frame1 = frame[face_location[0]:face_location[2],face_location[3]:face_location[1],:]
            img_to_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
            img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
            frame1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

            frame[face_location[0]:face_location[2],face_location[3]:face_location[1],:] = frame1  
        ###################################################
        
        faces = []
        
        # load classifier
        pickle_in = open(self.base_dir+"knn_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open(self.base_dir+"distance_thresholds.pickle","rb")
        distance_thresholds = pickle.load(pickle_in)
        pickle_in.close()
        
        # encode faces to be fed to classifier for prediction
        face_encodings = self.getFaceEncoding(frame, face_locations)
        
        # Find all facial features in all the faces in the image
        face_landmarks_list = self.getFacialLandmarks(frame, face_locations)
            
        # loop through face encodings and face boundary boxes
        for (top, right, bottom, left), face_encoding, face_landmark in zip(face_locations, face_encodings, face_landmarks_list):
            
            # get face pose
            face_pose = self.getFacePose(frame, top, right, bottom, left, face_landmark)
            
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
            cv2.rectangle(frame, (left, bottom), (right, bottom+40), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, str(avg_distance), (left + 6, bottom+37), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            #prob_dist = min(((0.55-avg_distance)/0.25)*100, 99)
            #total_prob = (confidence + prob_dist/100)/2
            
            #if (total_prob>=self.front_threshold and face_pose=='front') or (total_prob>=self.side_threshold and face_pose=='side'):
            if (face_pose=='front') and (avg_distance <= min(distance_thresholds[dist_name[0][0]]+self.knn_distance_threshold_margin, self.knn_distance_threshold) and confidence>=self.classifier_threshold):
                name = dist_name[0][0]
            elif (face_pose=='side') and ((avg_distance <= min(distance_thresholds[dist_name[0][0]]+self.knn_distance_threshold_margin, self.knn_distance_threshold) and confidence>=self.classifier_threshold) or (avg_distance <= min(distance_thresholds[dist_name[0][0]]+self.knn_distance_threshold_margin+0.02, self.knn_distance_threshold) and confidence>=self.classifier_threshold+0.05)):
                name = dist_name[0][0]
            faces.append((name, confidence))
            
            #print("[INFO] Face Pose:", face_pose)
            #print("[INFO] Distance Threshold:", distance_thresholds[dist_name[0][0]])
            #print("[INFO] Average Distance:", avg_distance)
            #print("[INFO] Probability:", confidence*100)
            #print("[INFO] Distance Probability:", prob_dist)
            #print("[INFO] Total Probability:", total_prob*100)
            #print("[INFO] Class:", dist_name[0][0], "\n")
            
            '''try:
                print("########################################################")
                for i in range(len(dist_name)):
                    print("[INFO] Average Distance:", round(sum(dist_name[i][1])/len(dist_name[i][1]),2))
                    print("[INFO] Probability:", (len(dist_name[i][1])/100))
                    print("[INFO] Class:", dist_name[i][0], "\n")
                print("########################################################")
            
            except:
                pass'''
        return face_encodings, face_landmarks_list, faces

    def saveDistanceThresholds(self, config):
        #frame = cv2.imread(os.path.join('raw_data','1239','WIN_20190125_16_50_20_Pro.jpg'))
        #user_details = face.getDistance(frame)
        users = os.listdir(config["user_images"])
        try:
            pickle_in = open(config['base_dir']+"distance_thresholds.pickle","rb")
            distance_thresholds = pickle.load(pickle_in)
            pickle_in.close()
        except:
            distance_thresholds = {}
        #"C:/Users/abhinav.jhanwar/Downloads/CASIA-WebFace/CASIA-WebFace/new_data"
        for user in tqdm(users):
            if user not in distance_thresholds.keys():
                trained_flag = False
                images = os.listdir(os.path.join(config["user_images"], user))
                #print("[INFO] User: %s"%user)
                for i, image in enumerate(images):
                    frame = cv2.imread(os.path.join(config["user_images"],user,image))
                    #frame = cv2.imread('raw_data/1242/WIN_20190221_16_31_37_Pro.jpg')
                    user_details = face.getDistance(frame)
                    if len(user_details)==1 and user_details[0][3] == 'front' and user_details[0][1]>0.9 and user_details[0][0]==user and user_details[0][2]<0.4:
                        trained_flag = True
                        distance_thresholds[user] = user_details[0][2]
                        break
                if trained_flag==False:
                    print("[INFO] Class: {0} does not have proper images\n".format(user))
                    distance_thresholds[user] = 0.1
                else:
                    print("[INFO] Class: {0} is successfull\n".format(user))
                
        pickle_out = open(config['base_dir']+"distance_thresholds.pickle","wb")
        pickle.dump(distance_thresholds, pickle_out)
        pickle_out.close()
        

if __name__=='__main__':
    
    ######## save distance threshold for each face #########################
    with open('config_V3.json', 'r') as outfile:  
        config = json.load(outfile)  
    face = faceDetection(config)
    face.saveDistanceThresholds(config)
    
    ########################################################################