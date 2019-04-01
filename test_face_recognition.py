# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:19:05 2019

@author: abhinav.jhanwar
"""


# face recognition modules
import cv2
# mathematical tools
import numpy as np
# time modules
from time import time
# models saving modules
import pickle
# image/video processing modules
import imutils
# json file handling module
import json
from faceDetection_V1 import faceDetection
import psutil, os


class webcamImageDetection:
   def __init__(self, config, camid):
        self.face = faceDetection(config)
        self.user_images = config['user_images']
        
   def startFaceDetection(self):
            
        # start frames capturing timer
        start_time = time()
        frame_counter = 0
        fps=0
        
        # start indefinite loop for video capturing
        for image in os.listdir(self.user_images):
            frame = cv2.imread(os.path.join(self.user_images, image))     
            
                
            face_locations = self.face.detectFace(frame)
            
            '''################################### histogram equalization
            for face_location in face_locations:
                #print("[INFO] Face Locations:", face_location)
                frame1 = frame[face_location[0]:face_location[2],face_location[3]:face_location[1],:]
                img_to_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
                img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
                frame1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

                frame[face_location[0]:face_location[2],face_location[3]:face_location[1],:] = frame1  
            ############################################################'''
            
            users = self.face.getFaceID(frame, face_locations)
            # loop through face encodings and face boundary boxes
            for (top, right, bottom, left), user in zip(face_locations, users):
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, user[0], (left+6, bottom+15), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "FPS: %s"%str(round(fps,2)), (20, 20), font, 0.5, (255, 255, 255), 1)
                
                if user[0]!='Unknown':
                    # put label for confidence
                    cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, str(round(user[1]*100,2))+'%', (left+6, top-3), font, 0.5, (255, 255, 255), 1)
                    
            # Display the resulting image
            cv2.imwrite('recognized_images/'+image, frame)
    
            
            # update the FPS counter
            frame_counter+=1
            fps = frame_counter / (time() - start_time)
            #print(fps)
         
        # stop the timer and display FPS information
        print("[INFO] elapsed time: {:.2f}".format(time() - start_time))
        print("[INFO] approx. FPS: {:.2f}".format(fps))
        
        
if __name__=='__main__':
    with open('config_V1.json', 'r') as outfile:  
        config = json.load(outfile)  
    face = webcamImageDetection(config, camid=0)
    face.startFaceDetection()