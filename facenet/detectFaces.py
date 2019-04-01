# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:07:37 2019

@author: abhinav.jhanwar
"""

# face recognition modules
import face_recognition
import cv2
# mathematical tools
import numpy as np
# time modules
from time import time
# models saving modules
import pickle
# image/video processing modules
from imutils.video import FPS
import imutils
# json file handling module
import json
import tensorflow as tf
from tensorflow.python.platform import gfile

class faceDetection:
    def __init__(self):
        # read configuration file
        with open('config.json', 'r') as outfile:  
            config = json.load(outfile)
        self.base_dir = config['base_dir']
        model_cfg = config['model_cfg']
        model_weights = config['model_weights']
        self.nms_threshold = config['nms_threshold']
        self.yolo_conf_threshold = config['yolo_conf_threshold']
        self.svm_threshold = config['svm_threshold']
        self.img_size = config['img_size']
        self.margin = config['margin']
    
        # define deep neural network parameters
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def load_model(self, model, input_map=None):
        print('Model filename: %s' % model)
        with gfile.FastGFile(model,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')

    ##########################################################################
    ############################ Live video detection ########################
    ##########################################################################
    def start_detection(self, camid=0):
            
        # connect to mqtt client
        if self.mqtt_flag == "True":
            client = mqtt.Client()
            client.connect(self.mqtt_client, port=1883, keepalive=0, bind_address="")
        
        # load classifier
        pickle_in = open(self.base_dir+"svm_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        # setup camera to capture video
        video_capture = cv2.VideoCapture(camid)
        
        # start frames capturing timer
        fps = FPS().start()
        
        # start tf sessions
        with tf.Graph().as_default():
            with tf.Session() as sess:
                
                self.load_model('20180402-114759.pb')
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                
                # start indefinite loop for video capturing
                while True:
                    # initialize data to be sent on mqtt server
                    data={}
                    # add time stamp
                    data['time'] = int(time())
                    # add cameraid/shelfid
                    data['cameraid'] = self.cameraid
             
                    # fetch camera frame
                    ret, frame = video_capture.read()
                    
                    '''ret=True
                    url='http://192.168.43.79:8086/shot.jpg'
                    imgResp=urllib.request.urlopen(url)
                    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
                    frame=cv2.imdecode(imgNp,-1) '''
                    
                    # validate if image is captured, else stop video capturing
                    if ret!=True:
                        print("\nCamera not detected")
                        video_capture.release()
                        cv2.destroyAllWindows()
                        return
                    
                    ##################################
                    # apply image processing
                    #######################################
                    #frame = imutils.resize(frame, height=200)
                    
                    # load model parameters
                    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416),
                                 [0, 0, 0], 1, crop=False)
                    self.net.setInput(blob)
                    
                    # fetch model predictions
                    layers_names = self.net.getLayerNames()
                    outs = self.net.forward([layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()])
                    
                    # fetch captured image dimensions
                    (frame_height, frame_width) = frame.shape[:2]
                    
                    # declare confidences, bounding boxes and face location bounding boxes list
                    confidences = []
                    boxes = []
                    face_locations = []
                    
                    # looping through grid cells
                    for out in outs:
                        # looping through detectors
                        for detection in out:
                            # fetch classes probability
                            scores = detection[5:]
                            # fetch class with maximum probability
                            class_id = np.argmax(scores)
                            # fetch maximum probability
                            confidence = scores[class_id]
                            # filter prediction based on threshold value
                            if confidence > self.yolo_conf_threshold:
                                # fetch validated bounding boxes
                                center_x = int(detection[0] * frame_width)
                                center_y = int(detection[1] * frame_height)
                                width = int(detection[2] * frame_width)
                                height = int(detection[3] * frame_height)
                                left = int(center_x - width / 2)
                                top = int(center_y - height / 2)
                                # add confidences and bounding boxes in list
                                confidences.append(float(confidence))
                                boxes.append([left, top, width, height])
                    
                    # perform non maximum suppression to remove overlapping images based on nms_threshold value           
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_conf_threshold,
                                               self.nms_threshold)
                    
                    # fetch legitimate face bounding boxes
                    for i in indices:
                        i = i[0]
                        box = boxes[i]
                        left = box[0]
                        top = box[1]
                        width = box[2]
                        height = box[3]
                        face_locations.append(np.array([top, left+width, top+height, left
                                     ]))
                    
                    # get face
                    det = face_locations[0]
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-self.margin/2, 0)
                    bb[1] = np.maximum(det[3]-self.margin/2, 0)
                    bb[2] = np.minimum(det[2]+self.margin/2, frame_height)
                    bb[3] = np.minimum(det[1]+self.margin/2, frame_width)
                    cropped = frame[bb[0]:bb[2], bb[1]:bb[3],:]
            
                    # resize image
                    scaled = imutils.resize(cropped, height=self.img_size)
            
                    # encode faces to be fed to classifier for prediction
                    images = np.zeros((1, self.img_size, self.img_size, 3))
                    scaled = cv2.resize(scaled, (self.img_size, self.img_size))
                    images[0,:,:,:] = scaled
                
                    feed_dict = {images_placeholder:images, phase_train_placeholder:False}
                    face_encodings = np.zeros((1, embedding_size))
                    face_encodings = sess.run(embeddings, feed_dict=feed_dict)
        
                    count=0
                    # loop through face encodings and face boundary boxes
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        face_encoding = [face_encoding]
                        # fetch probability distribution of predicted classes
                        probabilities = classifier.predict_proba(face_encoding)[0]
                        
                        # fetch maximum probability value
                        confidence = max(probabilities)
                        print(classifier.classes_[np.argmax(probabilities)], confidence, max(confidences))
                        
                        # set name as unknown if confidence is lower than threshold value
                        name = "Unknown"
                        if confidence>self.svm_threshold:
                            # fetch class with maximum probability
                            name = classifier.classes_[np.argmax(probabilities)]
                        
                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
                        
                        if name!='Unknown':
                            # put label for confidence
                            cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
                            cv2.putText(frame, str(round(confidence*100,2))+'%', (left + 6, top-3), font, 0.5, (255, 255, 255), 1)
                            
                            # prepare data as per mqtt requirements
                            data['userid'] = int(name)
                            data['x']=int(left)
                            data['y']=int(top)
                            data['confidence']=int(round(confidence*100,2))
                            data['No']=int(count)
                            
                            # convert dictionary to json data
                            json_data = json.dumps(data)
                            
                            if self.mqtt_flag == "True":
                                # publish data to mqtt server
                                (code, number) = client.publish(self.queue, json_data)
                                if code!=0:
                                     client.connect(self.mqtt_client, port=1883, keepalive=0, bind_address="")
                                     (code, number) = client.publish(self.queue, json_data)    
                            #print(json_data)
                        count+=1
            
                    # Display the resulting image
                    cv2.imshow('Face Recognition', frame)
            
                    # Hit 'q' on the keyboard to quit!
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if self.mqtt_flag == "True":
                            client.disconnect()
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

if __name__=='__main__': 
    
    # intialize faceDetection object
    face = faceDetection()
    
    # start face detection
    face.start_detection()
