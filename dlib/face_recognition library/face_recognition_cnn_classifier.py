# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:21:35 2018

@author: abhinav.jhanwar
"""

''' hog + cnn or cnn + cnn '''

import face_recognition
import cv2
from imutils.video import FPS
import pickle
from keras.models import load_model
import numpy as np
import pandas as pd

# load pre-traind encodings
pickle_in = open("SavedModels/known_face_encodings.pickle","rb")
known_face_encodings = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("SavedModels/known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("SavedModels/y_train_le.pickle","rb")
le = pickle.load(pickle_in)
pickle_in.close()

classifier = load_model("SavedModels/model.h5")

conf_threshold = 0.5

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

fps = FPS().start()
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_encoding = [face_encoding]
        prediction = list(classifier.predict(pd.DataFrame(face_encoding))[0])
        confidence = max(prediction)
        
        name = "Unknown"
        print(prediction)

        # If a match was found in known_face_encodings, just use the first one.
        if confidence>conf_threshold:
            name = le.classes_[np.argmax(prediction)]
            confidence = str(round(confidence*100,2))+'%'
            #print(name, "Distance: ", face_distances[first_match_index])

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
        
        # label for confidence
        cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, str(confidence), (left + 6, top-3), font, 0.5, (255, 255, 255), 1)


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