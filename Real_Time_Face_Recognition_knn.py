# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:28:47 2019

@author: abhinav.jhanwar
"""

# face recognition modules
import face_recognition
import cv2
# mathematical tools
import numpy as np
# time modules
from time import time
# models saving modules
import pickle
# image/video processing modules
from imutils.video import FPS
import imutils
# json file handling module
import json
from collections import defaultdict
from sys import getsizeof
import psutil, os

class faceDetection:
    
    # declare slots to reduce memory usage
    __slots__ = 'model_weights', 'base_dir', 'nms_threshold', 'yolo_conf_threshold',\
    'classifier_threshold', 'knn_distance_threshold', 'net', 'known_face_names', 'model_weights', 'model_cfg'
    
    def __init__(self):
        # read configuration file
        with open('config.json', 'r') as outfile:  
            config = json.load(outfile)
        self.base_dir = config['base_dir']
        self.model_cfg = config['model_cfg']
        self.model_weights = config['model_weights']
        self.nms_threshold = config['nms_threshold']
        self.yolo_conf_threshold = config['yolo_conf_threshold']
        self.classifier_threshold = config['classifier_threshold']
        self.knn_distance_threshold = config['knn_distance_threshold']
    
        # define deep neural network parameters
        self.net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        pickle_in = open(self.base_dir+"known_face_names.pickle","rb")
        self.known_face_names = pickle.load(pickle_in)
        pickle_in.close()
        
    ##########################################################################
    ############################ Live video detection ########################
    ##########################################################################
    def start_detection(self, camid=0):
            
        # load classifier
        pickle_in = open(self.base_dir+"knn_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        
        # setup camera to capture video
        video_capture = cv2.VideoCapture(camid)
        
        # start frames capturing timer
        fps = FPS().start()
        
        # start indefinite loop for video capturing
        while True:
            # fetch camera frame
            ret, frame = video_capture.read()
            
            '''ret=True
            url='http://192.168.43.79:8086/shot.jpg'
            imgResp=urllib.request.urlopen(url)
            imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
            frame=cv2.imdecode(imgNp,-1) '''
            
            # validate if image is captured, else stop video capturing
            if ret!=True:
                print("\nCamera not detected")
                video_capture.release()
                cv2.destroyAllWindows()
                return
            
            ##################################
            # apply image processing
            #######################################
            #frame = imutils.resize(frame, height=200)
            
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
                face_locations.append(np.array([top, left+width, top+height, left
                             ]))
            
            gamma=2
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                   for i in np.arange(0, 256)]).astype("uint8")
            frame1 = cv2.LUT(frame, table)
            
            img_to_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
            img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
            frame1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
            
            # encode faces to be fed to classifier for prediction
            face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=1)
            
            count=0
            # loop through face encodings and face boundary boxes
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_encoding = [face_encoding]
                #probabilities = classifier.predict_proba(face_encoding)
                
                predictions = classifier.kneighbors(face_encoding, n_neighbors=100)#classifier.n_neighbors)
     
                dist_name = defaultdict(list)
                [dist_name[self.known_face_names[key]].append(value) for value, key in zip(predictions[0][0], predictions[1][0])]
                
                # sort dictionary based on number of values
                dist_name = sorted(dist_name.items(), key=lambda item: len(item[1]), reverse=True)
                
                # fetch average distance of top class from given image
                avg_distance = round(sum(dist_name[0][1])/len(dist_name[0][1]),2)
                confidence = (len(dist_name[0][1])/100)#classifier.n_neighbors)
                
                name = "Unknown"
                if (avg_distance <= self.knn_distance_threshold and confidence>=self.classifier_threshold) or (avg_distance <= self.knn_distance_threshold+0.08 and confidence>=0.8):
                    name = dist_name[0][0]
                    
                # fetch maximum probability value
                #confidence = max(probabilities[0])
                #print(classifier.classes_[np.argmax(probabilities)], confidence, max(confidences))
                
                # set name as unknown if confidence is lower than threshold value
                #name = "Unknown"
                #if confidence>self.classifier_threshold:
                #    # fetch class with maximum probability
                #    name = classifier.classes_[np.argmax(probabilities)]
                
                #   print("\n[INFO] Probabilities:", probabilities[0], probability*100, avg_distance, "\nName:", name, "Name1:", dist_name[0][0])
                print("[INFO] Average Distance:", avg_distance)
                print("[INFO] Probability:", confidence)
                print("[INFO] Class:", dist_name[0][0], "\n")
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name+','+dist_name[0][0], (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
                
                if name!='Unknown':
                    # put label for confidence
                    cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, str(round(confidence*100,2))+'%', (left + 6, top-3), font, 0.5, (255, 255, 255), 1)
                    
                count+=1
    
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
        cv2.destroyAllWindows()
        video_capture.release()

if __name__=='__main__': 
    
    # intialize faceDetection object
    face = faceDetection()
    # start face detection
    face.start_detection()
