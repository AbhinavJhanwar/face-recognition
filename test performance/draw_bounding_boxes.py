# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:23:41 2019

@author: abhinav.jhanwar
"""

import face_recognition
import os
import numpy as np
from tqdm import tqdm
import cv2
import time
import logging

def evaluate(face_detector, images_loc, images):    
    if face_detector == 'yolo':
        model_cfg = "yolov3-face.cfg"
        model_weights = "yolov3-wider_16000.weights"
        net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        conf_threshold = 0.7
        nms_threshold = 0.3
     
    elif face_detector == 'opencv_resnet':
        modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        conf_threshold = 0.7
        nms_threshold = 0.3
             
    # Evaluate face detector and iterate it over dataset
    for i, image in tqdm(enumerate(images)):
        image_data = cv2.imread(os.path.join(images_loc, image))
        
        if face_detector == 'dlib_hog':
            face_pred = np.array(face_recognition.face_locations(image_data, model="hog"))
            for i, value in enumerate(face_pred):
                face_pred[i][0], face_pred[i][1], face_pred[i][2], face_pred[i][3] = face_pred[i][3], face_pred[i][0], face_pred[i][1], face_pred[i][2]
                
        elif face_detector == 'dlib_cnn':
            face_pred = np.array(face_recognition.face_locations(image_data, model="cnn"))
            for i, value in enumerate(face_pred):
                face_pred[i][0], face_pred[i][1], face_pred[i][2], face_pred[i][3] = face_pred[i][3], face_pred[i][0], face_pred[i][1], face_pred[i][2]
               
        elif face_detector == 'yolo':
            face_pred = []
            blob = cv2.dnn.blobFromImage(image_data, 1 / 255, (416, 416),
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
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                face_pred.append(np.array([left, top, left + width,
                             top + height]))
            
        elif face_detector == 'opencv_resnet':
            face_pred = []
            blob = cv2.dnn.blobFromImage(image_data, 1.0,
	                                        (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            (h, w) = image_data.shape[:2]
            confidences = []
            boxes = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    # startX, startY, endX, endY
                    confidences.append(float(confidence))
                    boxes.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
            
            
            try:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                       nms_threshold)
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    left = box[0]
                    top = box[1]
                    width = box[2]
                    height = box[3]
                    face_pred.append(np.array([left, top, left + width,
                                 top + height]))
                    
            except:
                for box in boxes:
                    left = box[0]
                    top = box[1]
                    width = box[2]
                    height = box[3]
                    face_pred.append(np.array([left, top, left + width,
                                 top + height]))
                
           
        # draw bounding boxes on all the faces in the image
        for i, pred in enumerate(face_pred):
            cv2.rectangle(image_data, (pred[0], pred[1]), (pred[2], pred[3]), (0, 0, 255), 2)
            
        # save image
        cv2.imwrite(os.path.join(face_detector, image), image_data)

if __name__=='__main__':
    # set images directory
    images_loc = "images"
    # list all the images names
    images = os.listdir(images_loc)
    
    # list all the detectors to be tested
    detector_list = [
            'opencv_resnet', 'yolo'#, 'dlib_hog', 'dlib_cnn', 'yolo'
        ]
    
    # evaluate each detector
    for detector in detector_list:
        face_detector = detector
        print("[INFO]", "processing", face_detector)
        start_time = time.time()
        if os.path.exists(detector):
            pass
        else:
            os.mkdir(detector)
        evaluate(face_detector, images_loc, images)
        inf_time = time.time() - start_time
        print("[INFO]", face_detector, "processed in", inf_time, "seconds")
        
    print("[INFO] PROCESS COMPLETE !!!")
        