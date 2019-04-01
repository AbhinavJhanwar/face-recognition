# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:01:43 2018

@author: abhinav.jhanwar
"""

''' hog + svm'''

import face_recognition
import cv2
from sklearn.svm import SVC
import os, glob
import numpy as np
from tqdm import tqdm
import pandas as pd

base_dir = "SavedModels/"

known_face_encodings=[]
known_face_names=[]

def train(verbose=True):
    # get name of person to be trained
    names = input("Write names of persons to be trained:\n").split(',')
    
    if len(names[0])>0:
        print("Encoding Images...\n")
        for name in tqdm(names):
            base = "known_people"
            name = name.strip()
            # find images of person to be trained
            base = os.path.join(base, name)
            for img_path in tqdm(glob.glob(os.path.join(base, "*"))):
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image, model="hog")
                if len(face_locations) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    known_face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=1)[0])
                    known_face_names.append(name)
        print("Encoding Completed!\n")
    
    else:
        print("Encoding Skipped!\n")
    
  
    # Create and train the SVM classifier
    svm_clf = SVC(kernel='linear', random_state=42, probability=True)
    svm_clf.fit(known_face_encodings, known_face_names)

    return svm_clf


# STEP 1: Train the KNN classifier and save it to disk
print("Training SVM classifier...")
classifier = train()
print("Training complete!")

base_data = pd.DataFrame()
confidence_threshold = 0.7

base_validate = "validate"
base_test = "tester"

predictions = list()
dirs = list()
confidences = list()

for img_path in tqdm(glob.glob(os.path.join(base_validate, "*.jpg"))):
    image = cv2.imread(img_path)
    
    face_locations = face_recognition.face_locations(image)
    if len(face_locations)!=1:
        print(len(face_locations), img_path, "not suitable")
        dirs.append(img_path)
        predictions.append("Not Defined")
        confidences.append(0)
        continue
    face_encodings = face_recognition.face_encodings(image, face_locations)
    probabilities = classifier.predict_proba(face_encodings)[0]
    confidence = max(probabilities)
    confidences.append(confidence)
        
    if confidence>confidence_threshold:
        name = classifier.classes_[np.argmax(probabilities)]
        predictions.append(name)
    else:
        name = "Unknown"
        predictions.append(name)
    dirs.append(img_path)
    print("\nProbability:", probabilities, "\nName:", name)

base_data['predicted'] = predictions
base_data['dir'] = dirs    
base_data['confidence'] = confidences     
base_data['test'] = [
                     "Unknown", 
                     "Chandler Bing", 
                     "Rachel Green", 
                     "Chandler Bing",
                     "Rachel Green",
                     "Rachel Green",
                     "Chandler Bing",
                     "Elon Musk",
                     "Elon Musk",
                     "Elon Musk"]

base_data['val'] = [
                     "Chandler Bing", 
                     "Unknown", 
                     "Elon Musk",
                     "Unknown",
                     "Elon Musk",
                     "Rachel Green", 
                     "Rachel Green",
                     "Rachel Green",
                     "Chandler Bing",
                     "Chandler Bing"
                     ]
print("Average Confidence:", sum([confidence for confidence in confidences if confidence>0.7])/len([confidence for confidence in confidences if confidence>0.7]))
# Rachel Green, Chandler Bing, Elon Musk