# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:14:31 2019

@author: abhinav.jhanwar
"""

# face recognition modules
import cv2
# mathematical tools
import numpy as np
# models saving modules
import pickle
# json file handling module
import json
import os, glob
from tqdm import tqdm
import imutils

with open('config.json') as outFile:
    config = json.load(outFile)
raw_data = config['raw_data']
model_cfg = config['model_cfg']
model_weights = config['model_weights']
output_dir = config['processed_data']
margin = config['margin']
yolo_conf_threshold = config['yolo_conf_threshold']
nms_threshold = config['nms_threshold']
img_size = config['img_size']

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# fetch all the classes to be trained
names=[]
for user in glob.glob(os.path.join(raw_data,'*')):
    names.append(user.split('\\')[-1])

pickle_out = open("classes.pickle","wb")  
pickle.dump(names, pickle_out)
pickle_out.close()

for name in tqdm(names):
    count=0
    # create output directories for each class
    if not os.path.exists(os.path.join(output_dir, name)):
        os.makedirs(os.path.join(output_dir, name))
        
    for img_path in glob.glob(os.path.join(raw_data+'/'+name, "*.jpg")):
        # read image
        frame = cv2.imread(img_path)
          
        # load model parameters
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416),
                     [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        
        # fetch model predictions
        layers_names =  net.getLayerNames()
        outs =  net.forward([layers_names[i[0] - 1] for i in  net.getUnconnectedOutLayers()])
        
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
                if confidence >  yolo_conf_threshold:
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
        indices = cv2.dnn.NMSBoxes(boxes, confidences,  yolo_conf_threshold,
                                    nms_threshold)
        
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
        if len(face_locations) != 1:
            # If there are no people (or too many people) in a training image, skip the image.
            print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
            
        else:
            det = face_locations[0]
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[3]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, frame_height)
            bb[3] = np.minimum(det[1]+margin/2, frame_width)
            
            # get face
            cropped = frame[bb[0]:bb[2], bb[1]:bb[3],:]
            
            # resize image
            scaled = cv2.resize(cropped, (img_size, img_size))
            
            # save cropped face
            cv2.imwrite(output_dir+'/'+name+'/'+name+str(count)+'.jpg', scaled) 
            #print(name, count)
        count+=1