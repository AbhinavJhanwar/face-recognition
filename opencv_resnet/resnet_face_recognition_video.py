# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:55:21 2018

@author: abhinav.jhanwar
"""

# modelFile- contains weights for the actual layers
# configFile - contains model architecture
import cv2
import numpy as np
import face_recognition
import pickle
from imutils.video import FPS

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

# set confidence threshold
conf_threshold = 0.6

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

fps = FPS().start()
while True:
    #t1 = time.time()
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if ret == False:
        print("frame not available")
        continue

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(rgb_frame, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # Find all the faces and face encodings in the frame of video
    face_locations = list()
    confidences = list()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf_threshold:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            location = tuple(box.astype("int"))
            face_locations.append((location[1], location[2], location[3], location[0]))
            confidences.append(confidence)
    
    #print("I found {} face(s) in this photograph.".format(len(face_locations)))
    #print(confidences)
    # num_jitters : how many times to upsample the image
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding, confidence in zip(face_locations, face_encodings, confidences):
        # See if the face is a match for the known face(s)
        # uses SVM model for classification
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #print(matches)
        name = "Unknown"
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            #print(name, "Confidence: {0}%".format(round((1-min(face_distances))*100, 2)))
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, str(confidence), (left + 6, top-3), font, 0.5, (255, 255, 255), 1)
    
    #t2 = time.time()
    #print("Time Taken: ", round(t2-t1, 2))
    
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
video_capture.release()
cv2.destroyAllWindows()

    