# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:33:46 2018

@author: abhinav.jhanwar
"""

# modelFile- contains weights for the actual layers
# configFile - contains model architecture
import cv2
import numpy as np

# load model from disk
print("[INFO] loading from model...")
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# load the input image and construct an input blob for the image and resize image to
# fixed 300x300 pixels and then normalize it
name = "Abhinav6.jpg"
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
conf_threshold = 0.2
# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    
    print(confidence)
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > conf_threshold:
        
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
         
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
        			(0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("dnn_face_detection.jpg", image)

    