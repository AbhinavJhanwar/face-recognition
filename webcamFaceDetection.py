# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:15:32 2019

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
from faceDetection_V3 import faceDetection
import psutil, os

class webcamImageDetection:
    __slots__ = 'camid', 'face'
    
    def __init__(self, config, camid):
        self.camid = camid
        self.face = faceDetection(config)
        
    def startFaceDetection(self):
            
        # setup camera to capture video
        video_capture = cv2.VideoCapture(self.camid)
        
        # start frames capturing timer
        start_time = time()
        frame_counter = 0
        fps=0
        
        # start indefinite loop for video capturing
        while True:
            
            # fetch camera frame
            ret, frame = video_capture.read()
            frame1 = frame.copy()
            
            # validate if image is captured, else stop video capturing
            if ret!=True:
                print("\n[INFO] Camera not detected")
                video_capture.release()
                cv2.destroyAllWindows()
                return
            
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
            
            # for version 1
            #users = self.face.getFaceID(frame1, face_locations)
            
            # for version 2 and 3
            face_encodings, face_landmarks_list, users = self.face.getFaceID(frame1, face_locations)
            
            #chin left_eyebrow right_eyebrow nose_bridge nose_tip left_eye right_eye top_lip bottom_lip
            count=0
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
                  
                count+=1
            
            # Display the resulting image
            cv2.imshow('Face Recognition', frame)
    
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # update the FPS counter
            frame_counter+=1
            fps = frame_counter / (time() - start_time)
            #print(fps)
         
        # stop the timer and display FPS information
        print("[INFO] elapsed time: {:.2f}".format(time() - start_time))
        print("[INFO] approx. FPS: {:.2f}".format(fps))
         
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        
        
if __name__=='__main__':
    with open('config_V3.json', 'r') as outfile:  
        config = json.load(outfile)  
    face = webcamImageDetection(config, camid=0)
    face.startFaceDetection()