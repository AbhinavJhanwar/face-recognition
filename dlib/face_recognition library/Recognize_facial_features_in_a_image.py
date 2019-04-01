# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:27:17 2018

@author: abhinav.jhanwar
"""

''' extract facial features and draw a line on them in image '''

from PIL import Image, ImageDraw
import face_recognition
# using facial landmarks

##############################################################################
########## Automatically locate the facial features of a person in an image #####################
#############################################################################

image = face_recognition.load_image_file("Jothi.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

# Show the picture
pil_image.show()
















