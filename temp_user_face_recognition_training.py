# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:57:59 2019

@author: abhinav.jhanwar
"""

# face recognition modules
import face_recognition
import cv2
# classifier modules
from sklearn import neighbors
# model evaluation modules
from sklearn.model_selection import GridSearchCV
# mathematical tools
import numpy as np
from tqdm import tqdm
# directory modules
import os
import glob
# models saving modules
import _pickle as pickle
# json file handling module
import json
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil
from distutils.util import strtobool
import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

class imageDataAugmentation:
    __slots__ = "datagen"
    def __init__(self, config):        
        # this is the augmentation configuration we will use for training
        self.datagen = ImageDataGenerator(
                rotation_range=config['rotation_range'],
                brightness_range=config['brightness_range'],
                shear_range=config['shear_range'],
                zoom_range=config['zoom_range'],
                channel_shift_range=config['channel_shift_range'],
                fill_mode=config['fill_mode'],
                horizontal_flip=strtobool(config['horizontal_flip']),
                vertical_flip=strtobool(config['vertical_flip']),
                rescale=config['rescale']
                )
        
    def generateData(self, config):
        # fetch all the folder names
        names = []
        dirs = os.listdir(config['directory'])
        for name in dirs:
            if name not in os.listdir(config['save_to_dir']):
                names.append(name)
        print('[INFO] generating data for following images -\n', names)
        
        for name in tqdm(names):
            # read first image for fetching the dimensions
            img = load_img(os.path.join(config['directory'], name, os.listdir(os.path.join(config['directory'], name))[0]))
            
            # remove any existing directories for the given name
            if os.path.exists(os.path.join(config['save_to_dir'], name)):
                continue
                
            # save the original images
            shutil.copytree(os.path.join(config['directory'], name), os.path.join(config['save_to_dir'], name))
        
            # this is a generator that will read pictures and generate batches of augmented image data
            generator = self.datagen.flow_from_directory(
                    directory = config['directory'],  # this is the image folders directory
                    target_size=(img.height, img.width),  # all images will be resized to size of actual images
                    color_mode=config["color_mode"],
                    class_mode=config["class_mode"],
                    batch_size=len(os.listdir(os.path.join(config['directory'], name))),
                    shuffle=strtobool(config["shuffle"]),
                    save_to_dir=os.path.join(config["save_to_dir"],name),
                    save_format=config["save_format"],
                    classes=[name]
                    )
                
            for i in tqdm(range(config['augmentBy'])):
                next(generator)
               
class faceTraining:
    __slots__ = 'fast_base_dir', 'user_images', 'nms_threshold', 'yolo_conf_threshold', 'margin', 'net', 'net2',\
                'names'
    def __init__(self, config):
       
        self.fast_base_dir = config['base_dir']
        model_cfg = config['model_cfg']
        model_weights = config['model_weights']
        self.user_images = config['ip_user_images']
        self.nms_threshold = config['nms_threshold']
        self.yolo_conf_threshold = config['yolo_conf_threshold']
        self.margin = config['margin_percent']
        
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        modelFile2 =  config['resnet_model']
        configFile2 =  config['resnet_cfg']
        self.net2 = cv2.dnn.readNetFromCaffe(configFile2, modelFile2)
        self.net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.names=[]
        for user in glob.glob(os.path.join(self.user_images,'*')):
            self.names.append(user.split('\\')[-1])
        
    def saveEncodings(self, verbose=True):
        # initialize a text file for saving images where faces are not found
        with open('waste_files.txt', 'w') as waste:
            waste.write("Images which are not suitable for training are-\n")
                    
            try:
                # load previously saved encodings
                pickle_in = open(self.fast_base_dir+"known_face_encodings.pickle","rb")
                known_face_encodings = pickle.load(pickle_in)
                pickle_in.close()
                
                pickle_in = open(self.fast_base_dir+"known_face_names.pickle","rb")
                known_face_names = pickle.load(pickle_in)
                pickle_in.close()
                # filter out faces which are already trained
                temp = []
                for name in self.names:
                    if name not in known_face_names:
                        temp.append(name)
                
                self.names = temp
                
            except:
                # declare encodings as empty
                known_face_encodings=[]
                known_face_names=[]
            
            print("[INFO] Encoding... ", self.names)
            if self.names != []:
                # looping through names to be trained
                for name in tqdm(self.names):
                    # load images of person to be trained
                    base = self.user_images
                    name = name.strip()
                    base = os.path.join(base, name)
                    # looping through images of person to be trained
                    for img_path in glob.glob(os.path.join(base, "*.jpg")):
                        # read image
                        image_data = cv2.imread(img_path)
                        
                        ###############
                        # check face using resnet
                        ###############
                        blob = cv2.dnn.blobFromImage(image_data, 1.0,
	                                        (300, 300), (104.0, 177.0, 123.0))
                        self.net2.setInput(blob)
                        detections = self.net2.forward()
                        (h, w) = image_data.shape[:2]
                        confidences = []
                        boxes = []
                        for i in range(0, detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.98:
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                box = box.astype("int")
                                # startX, startY, endX, endY
                                confidences.append(float(confidence))
                                boxes.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
            
                        if len(boxes)>0:
                            pass
                        else:
                           waste.write(img_path+"\n")
                           print("[INFO] Image {} not suitable for training: Resnet filtered out".format(img_path))
                           continue
                       
                        ###############
                        # face detection using yolo
                        ###############
                        # load model parameters
                        blob = cv2.dnn.blobFromImage(image_data, 1 / 255, (416, 416),
                                         [0, 0, 0], 1, crop=False)
                        self.net.setInput(blob)
                        # fetch predictions from model/network
                        layers_names = self.net.getLayerNames()
                        outs = self.net.forward([layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()])
                        # fetch size of image
                        (frame_height, frame_width) = image_data.shape[:2]
                        # declare overall confidence list
                        confidences = []
                        # declare bounding boxes list
                        boxes = []
                        face_locations = []
                        # looping through model predictions/ predictions for each grid cell
                        for out in outs:
                            # looping through detectors outputs for grid cell
                            for detection in out:
                                # fetch classifier probabilities for different classes
                                scores = detection[5:]
                                # fetch maximum probabilty class
                                class_id = np.argmax(scores)
                                # define confidence as maximum probability
                                confidence = scores[class_id]
                                # filter predictions based on confidence threshold
                                if confidence > self.yolo_conf_threshold:
                                    # fetch bounding box dimensions
                                    center_x = int(detection[0] * frame_width)
                                    center_y = int(detection[1] * frame_height)
                                    width = int(detection[2] * frame_width)
                                    height = int(detection[3] * frame_height)
                                    left = int(center_x - width / 2)
                                    top = int(center_y - height / 2)
                                    # append confidence in confidences list
                                    confidences.append(float(confidence))
                                    # append bounding box in bounding boxes list
                                    boxes.append([left, top, width, height])
                        
                        # perform non maximum suppression of overlapping images
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_conf_threshold,
                                                   self.nms_threshold)
                        
                        # fetch faces bounding boxes
                        for i in indices:
                            i = i[0]
                            box = boxes[i]
                            left = box[0]
                            top = box[1]
                            width = box[2]
                            height = box[3]
                            face_locations.append(np.array([top, left+width+(width*self.margin//100), top+height, left-(width*self.margin//100)
                             ]))

                        if len(face_locations) != 1:
                            waste.write(img_path+"\n")
                            # If there are no people (or too many people) in a training image, skip the image.
                            if verbose:
                                print("[INFO] Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
                        else:
                            for face_location in face_locations:
                                if min(face_location)<0:
                                    print("[INFO] Image {} not suitable for training: Face is not in Boundary of Image".format(img_path))
                                else:
                                    ######################################################
                                    # histogram equalization
                                    frame1 = image_data[face_location[0]:face_location[2],face_location[3]:face_location[1],:]
                                    img_to_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
                                    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
                                    frame1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
                        
                                    image_data[face_location[0]:face_location[2],face_location[3]:face_location[1],:] = frame1  
                                    ###################################################
                                    # Add face encoding for current image to the training set
                                    known_face_encodings.append(face_recognition.face_encodings(image_data, known_face_locations=face_locations, num_jitters=20)[0])
                                    known_face_names.append(name)
                
                    # save the encodings after every iteration of distinct class
                    pickle_out = open(self.fast_base_dir+"known_face_names.pickle","wb")
                    pickle.dump(known_face_names, pickle_out)
                    pickle_out.close()
                    pickle_out = open(self.fast_base_dir+"known_face_encodings.pickle","wb")
                    pickle.dump(known_face_encodings, pickle_out)
                    pickle_out.close()
                    print("[INFO]: %s saved!"%name)
                
            else:
                print("Encoding Skipped!\n")
    
    def getEncodedNames(self):
        pickle_in = open(self.fast_base_dir+"known_face_names.pickle","rb")
        known_face_names = pickle.load(pickle_in)
        pickle_in.close()
        return known_face_names
        
    def trainClassifier(self, optimize=False):
        # load dataset
        pickle_in = open(self.fast_base_dir+"known_face_encodings.pickle","rb")
        known_face_encodings = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open(self.fast_base_dir+"known_face_names.pickle","rb")
        known_face_names = pickle.load(pickle_in)
        pickle_in.close()
        
        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=-1)
        
        if optimize==True:
            # GridSearch for parameters optimization
            param_grid = {
                    'n_neighbors': list(range(1,51))
                    }
            grid = GridSearchCV(knn_clf, param_grid, verbose=3, n_jobs=10, cv=10)
            # fit model
            grid.fit(known_face_encodings, known_face_names)
            print("[INFO] Parameters selected for model:", grid.best_estimator_)
            print("[INFO] Score:", grid.best_score_)
            
        else:
            grid = knn_clf
            # fit model
            grid.fit(known_face_encodings, known_face_names)
                
        # save classifier
        pickle_out = open(self.fast_base_dir+"knn_clf.pickle","wb")
        pickle.dump(grid, pickle_out)
        pickle_out.close()
        knn_clf.fit(known_face_encodings, known_face_names)
        #print("[INFO] %d Images trained and saved successfully"%len(known_face_names))
        
class faceDetection:
    __slots__ = 'fast_base_dir', 'classifier_threshold', 'knn_distance_threshold', 'known_face_names',\
                'knn_distance_threshold_margin', 'directory'
    
    def __init__(self, config):
        self.fast_base_dir = config['base_dir']
        self.classifier_threshold = config['classifier_threshold']
        self.knn_distance_threshold = config['knn_distance_threshold']
        self.directory = config['directory']
        #self.knn_distance_threshold_margin = config['knn_distance_threshold_margin']
        
    def getFacePose(self, frame, top, right, bottom, left, face_landmark):
        left_eye = ((face_landmark['left_eye'][2][0]+face_landmark['left_eye'][1][0])//2, face_landmark['left_eye'][2][1])
        right_eye = ((face_landmark['right_eye'][2][0]+face_landmark['right_eye'][1][0])//2, face_landmark['right_eye'][2][1])
        nose_tip = face_landmark['nose_tip'][2]
            
        frame_length = right-left
        # left_eye-right_eye, left_eye-nose_tip, right_eye-nose_tip
        distances = list(map(lambda x: abs(x[0][0]/frame_length-x[1][0]/frame_length), [(left_eye, right_eye), (left_eye, nose_tip), (right_eye, nose_tip)]))
        #print(distances)
        cv2.line(frame, left_eye, left_eye, (255, 255, 255), 2)
        cv2.line(frame, right_eye, right_eye, (255, 255, 255), 2)
        cv2.line(frame, nose_tip, nose_tip, (255, 255, 255), 2)
        #print('[INFO]', distances, '\n')
        font = cv2.FONT_HERSHEY_DUPLEX
        if (distances[0]<0.36 and distances[1]<0.11) or (distances[0]<0.36 and distances[2]<0.17):
            cv2.putText(frame, "SIDE POSE", (20, 40), font, 0.5, (255, 255, 255), 1)
            return 'side'
        else:
           cv2.putText(frame, 'FRONT POSE', (20, 40), font, 0.5, (255, 255, 255), 1)
           return 'front'
    
    def getFaceID(self, frame, top, right, bottom, left, face_encoding, face_landmarks):
        
        faces = []
        
        # load saved models
        pickle_in = open(self.fast_base_dir+"known_face_names.pickle","rb")
        self.known_face_names = pickle.load(pickle_in)
        pickle_in.close()
        
        # load classifier
        pickle_in = open(self.fast_base_dir+"knn_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        face_encoding = [face_encoding]
        # fetch probability distribution of predicted classes
        predictions = classifier.kneighbors(face_encoding, n_neighbors=10)
 
        dist_name = defaultdict(list)
        [dist_name[self.known_face_names[key]].append(value) for value, key in zip(predictions[0][0], predictions[1][0])]
        
        # sort dictionary based on number of values
        dist_name = sorted(dist_name.items(), key=lambda item: len(item[1]), reverse=True)
       
        # fetch average distance of top class from given image
        avg_distance = round(sum(dist_name[0][1])/len(dist_name[0][1]),2)
        confidence = (len(dist_name[0][1])/10)
        
        name = "Unknown"
        if (avg_distance <= self.knn_distance_threshold and confidence>=self.classifier_threshold) or (avg_distance <= self.knn_distance_threshold+0.06 and confidence>=0.9):
            name = dist_name[0][0]
        
        faces.append((name, confidence))
        
        print("[INFO] Average Distance:", avg_distance)
        print("[INFO] Probability:", confidence*100)
        print("[INFO] Class:", dist_name[0][0], "\n")
        
        return faces
    
    def trainFace(self):
        
        # read configuration
        with open('fastConfig_V1.json', 'r') as config:
            config = json.load(config)
        
        # generate data set
        data = imageDataAugmentation(config)
        data.generateData(config)
    
        # define class object
        face = faceTraining(config)
        
        # encode names
        face.saveEncodings(verbose=True)
        
        # start training and save model
        face.trainClassifier(optimize=False)
        
    def savePictures(self, images, userID):
        
        # create a directory to save images of new user
        if not os.path.exists(os.path.join(self.directory, userID)):
            os.mkdir(os.path.join(self.directory, userID))
        
        initial_images = len(os.listdir(os.path.join(self.directory, userID)))+1
        for i, frame in enumerate(images):
            cv2.imwrite(os.path.join(self.directory, userID)+"/{0}_{1}.jpg".format(userID, initial_images+i), frame)
        
        print("[INFO] CLICKED PICTURES OF NEW USER:", userID)
        
if __name__ == '__main__':
    # read configuration
    with open('fastConfig_V1.json', 'r') as config:
        config = json.load(config)
    
    # generate data set
    data = imageDataAugmentation(config)
    data.generateData(config)

    # define class object
    face = faceTraining(config)
    
    # encode names
    face.saveEncodings(verbose=True)
    
    # start training and save model
    face.trainClassifier(optimize=False)
    