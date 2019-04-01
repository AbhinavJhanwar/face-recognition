# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:34:33 2018

@author: abhinav.jhanwar
"""


import face_recognition
import cv2
from imutils.video import FPS


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
Abhinav_image = face_recognition.load_image_file("known_people/Abhinav.jpg")
Abhinav_face_encoding = face_recognition.face_encodings(Abhinav_image)[0]

# Load a second sample picture and learn how to recognize it.
jobs_image = face_recognition.load_image_file("known_people/steve-jobs.jpg")
jobs_face_encoding = face_recognition.face_encodings(jobs_image)[0]

# Load a fourth sample picture and learn how to recognize it.
Jothi_image = face_recognition.load_image_file("known_people/Jothi.jpg")
Jothi_face_encoding = face_recognition.face_encodings(Jothi_image)[0]

# Create arrays of known face encodings and their names
# Create arrays of known face encodings and their names
known_face_encodings = [
    Abhinav_face_encoding,
    jobs_face_encoding,
    Jothi_face_encoding
]
known_face_names = [
    "Abhinav Jhanwar",
    "Steve Jobs",
    "Jothi"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

fps = FPS().start()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    # number_of_times_to_upsample: higher number to find smaller faces
    # model = "cnn" slower but accurate
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    # Loop through each face in this frame of video
    for (top, right, bottom, left) in face_locations:
       
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        
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