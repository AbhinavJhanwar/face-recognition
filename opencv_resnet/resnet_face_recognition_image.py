# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:33:46 2018

@author: abhinav.jhanwar
"""

# modelFile- contains weights for the actual layers
# configFile - contains model architecture
import cv2
import numpy as np
import face_recognition
import pickle

# load pre-traind encodings
pickle_in = open("known_face_encodings.pickle","rb")
known_face_encodings = pickle.load(pickle_in)

pickle_in = open("known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)

# load model from disk
print("[INFO] loading from model...")
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# load the input image and construct an input blob for the image and resize image to
# fixed 300x300 pixels and then normalize it
name = "outside_000001.jpg"
name = "yoloface/WIN_20181130_10_54_33_Pro.jpg"
image = cv2.imread(name)
(h, w) = image.shape[:2]
# 1.0 is scalefactor
# next (300, 300) is spatial size that Convolutional Neural Network expects
# last values are mean subtraction values in tuple and they are RGB means
# for more details check tutorial - Face detection with OpenCV and Deep Learning from image-part 1
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# set confidence threshold
conf_threshold = 0.3

# define face location list
face_locations = list()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    #print(confidence)
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > conf_threshold:
        
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        location = tuple(box.astype("int"))
        face_locations.append((location[1], location[2], location[3], location[0]))

print("I found {} face(s) in this photograph.".format(len(face_locations)))
    
face_encodings = face_recognition.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
    name = "Unknown"
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        print(name, "Confidence: {0}%".format(round((1-min(face_distances))*100, 2)))
    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    # Draw a label with a name below the face
    cv2.rectangle(image, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)




cv2.imwrite("dnn_face_detection.jpg", image)

    