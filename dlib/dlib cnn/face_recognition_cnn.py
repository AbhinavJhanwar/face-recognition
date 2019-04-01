# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:02:37 2018

@author: abhinav.jhanwar
"""

import cv2
import dlib

# load input image
#image = cv2.imread("input.jpg")
image = cv2.imread("outside_000001.jpg")

if image is None:
    print("Could not read input image")
    exit()
    

########################
########## HOG ########
##########################
# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# apply face detection (hog)
faces_hog = hog_face_detector(image, 1)

# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    
    
################################
########### CNN ##############
###############################

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# apply face detection (cnn)
faces_cnn = cnn_face_detector(image, 1)

# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

     # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

# write at the top left corner of the image
# for color identification
img_height, img_width = image.shape[:2]
cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

# display output image
#cv2.imshow("face detection with dlib", image)
#cv2.waitKey()

# save output image 
cv2.imwrite("face_detection.png", image)

# close all windows
#cv2.destroyAllWindows()