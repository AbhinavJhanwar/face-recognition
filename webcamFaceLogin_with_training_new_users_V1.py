# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:44:32 2019

@author: abhinav.jhanwar
"""

# face recognition modules
import cv2
# mathematical tools
import numpy as np
# time modules
from time import time
# image/video processing modules
from imutils.video import FPS
import imutils
# json file handling module
import json
from faceDetection_V3 import faceDetection
# import secondary face detection model
from temp_user_face_recognition_training import faceDetection as tempfaceDetection
import os

class webcamImageDetection:
    __slot__ = "camid", "face", "face1", "video_capture"
    def __init__(self, config, camid, config1):
        self.camid = camid
        self.face = faceDetection(config)
        self.face1 = tempfaceDetection(config1)
        
    def startFaceDetection(self):
            
        # setup camera to capture video
        self.video_capture = cv2.VideoCapture(self.camid)
        
        # start frames capturing timer
        fps = FPS().start()
        
        # start indefinite loop for video capturing
        while True:
            
            # fetch camera frame
            ret, frame = self.video_capture.read()
            frame1 = frame.copy()
            
            # validate if image is captured, else stop video capturing
            if ret!=True:
                print("\n[INFO] Camera not detected")
                self.video_capture.release()
                cv2.destroyAllWindows()
                return
            
            # fetch face location
            face_locations = self.face.detectFace(frame)
            # fetch user details
            face_encodings, face_landmarks_list, users = self.face.getFaceID(frame1, face_locations)
            
            count=0
            # loop through face encodings and face boundary boxes
            for (top, right, bottom, left), user, face_encoding, face_landmarks in zip(face_locations, users, face_encodings, face_landmarks_list):
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                if user[0]!='Unknown':
                    # put label for confidence
                    cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, str(round(user[1]*100,2))+'%', (left + 6, top-3), font, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, user[0], (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
                    
                else:
                    print("[INFO] Loading secondary model")
                    # make prediction using secondary model
                    # recognize face
                    user = self.face1.getFaceID(frame1, top, right, bottom, left, face_encoding, face_landmarks)[0]
                    if user[0] == 'Unknown':
                        print("[INFO] Face not available in secondary model")
                        cv2.destroyAllWindows()
                        # train the unknown face
                        # assign user a temporary ID
                        #userID = input("[INPUT REQUIRED] Please enter user id: ")
                        userID = str(2000+len(os.listdir(self.face1.directory)))
                        # take pictures of user and verify he is not recognized
                        flag, user, images = self.verifyUser()
                        # if user is not recognized then train the user
                        if flag==False:
                            print("[INFO] Please wait while your face is being trained...")
                            self.face1.savePictures(images, userID)
                            self.face1.trainFace()
                            print("[INFO] Training Completed. Starting Recognition process...")
                            # recognize again
                            user = self.face1.getFaceID(frame1, top, right, bottom, left, face_encoding, face_landmarks)[0]
                            if user[0] == 'Unknown':
                                continue
                    
                    cv2.putText(frame, user[0], (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, str(round(user[1]*100,2))+'%', (left + 6, top-3), font, 0.5, (255, 255, 255), 1)
                    
                count+=1
                    
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
        self.video_capture.release()
        cv2.destroyAllWindows()
        
    def verifyUser(self):
        
        images = []
        for i in range(5):
            # fetch camera frame
            ret, frame = self.video_capture.read()
            
            # validate if image is captured, else return
            if ret!=True:
                print("\n[INFO] Camera not detected")
                return
            
            cv2.imshow("Picture %d of 5"%(i+1), frame)
            cv2.waitKey(1)
            
            # fetch face location
            face_locations = self.face.detectFace(frame)
            # fetch user details
            face_encodings, face_landmarks_list, users = self.face.getFaceID(frame, face_locations)
            for (top, right, bottom, left), user, face_encoding, face_landmarks in zip(face_locations, users, face_encodings, face_landmarks_list):
                if user[0] != 'Unknown':
                    cv2.destroyAllWindows()
                    return True, user, None
                else:
                    user = self.face1.getFaceID(frame, top, right, bottom, left, face_encoding, face_landmarks)[0]
                    if user[0] != 'Unknown':
                        cv2.destroyAllWindows()
                        return True, user, None
        
            images.append(frame)
            
        cv2.destroyAllWindows()
        return False, None, images
        
if __name__=='__main__':
    with open('config_V3.json', 'r') as outfile:  
        config = json.load(outfile) 
        
    with open('fastConfig_V1.json', 'r') as outfile:  
        config1 = json.load(outfile) 
        
    fast_detection = webcamImageDetection(config, camid=0, config1=config1)
    fast_detection.startFaceDetection()