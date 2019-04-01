# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:49:14 2018

@author: abhinav.jhanwar
"""

import face_recognition
import cv2
from imutils.video import FPS

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Load a sample picture and learn how to recognize it.
Abhinav_image = face_recognition.load_image_file("known_people/Abhinav.jpg")
Abhinav_face_encoding = face_recognition.face_encodings(Abhinav_image)[0]

# Load a second sample picture and learn how to recognize it.
jobs_image = face_recognition.load_image_file("known_people/steve-jobs.jpg")
jobs_face_encoding = face_recognition.face_encodings(jobs_image)[0]

Albert_image = face_recognition.load_image_file("known_people/Albert-Einstein.jpg")
Albert_face_encoding = face_recognition.face_encodings(Albert_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    Abhinav_face_encoding,
    jobs_face_encoding,
    Albert_face_encoding
]
known_face_names = [
    "Abhinav Jhanwar",
    "Steve Jobs",
    "Albert Einstein"
]

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


fps = FPS().start()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    # number_of_times_to_upsample: higher number to find smaller faces
    # model = "cnn" slower but accurate
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    # num_jitters : how many times to upsample the image
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            #print(name, "Distance: ", face_distances[first_match_index])

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)

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