# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:35:57 2018

@author: abhinav.jhanwar
"""


import face_recognition
import cv2
from sklearn import neighbors
from collections import defaultdict
import os, glob
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
    
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='auto', weights='distance', n_jobs=-1)
    knn_clf.fit(known_face_encodings, known_face_names)
    return knn_clf


# STEP 1: Train the KNN classifier and save it to disk
print("Training KNN classifier...")
classifier = train()
print("Training complete!")


# STEP 2: Predictions
base_data = pd.DataFrame()
distance_threshold=0.5

base_validate = "validate"
base_test = "tester"

predictions = list()
dirs = list()
distances = list()

for img_path in tqdm(glob.glob(os.path.join(base_validate, "*.jpg"))):
    image = cv2.imread(img_path)
    
    face_locations = face_recognition.face_locations(image)
    if len(face_locations)!=1:
        print(len(face_locations), img_path, "not suitable as more than 1 face")
        dirs.append(img_path)
        predictions.append("Not Defined")
        distances.append(1)
        continue
    face_encodings = face_recognition.face_encodings(image, face_locations)
    closest_distances = classifier.kneighbors(face_encodings, n_neighbors=2)
     
    dist_name = defaultdict(list)
    [dist_name[known_face_names[key]].append(value) for value, key in zip(closest_distances[0][0], closest_distances[1][0])]
    # get max frequent name in dict
    max_count=0
    for key in dist_name.keys():
        count = len(dist_name[key])
        if count>max_count:
            max_count=count
            avg_name = key
        
    avg_distance = round(sum(dist_name[avg_name])/len(dist_name[avg_name]),2)
    if avg_distance <= distance_threshold:
        name = avg_name
        distance = avg_distance
        predictions.append(name)
        distances.append(avg_distance)
    else:
        name = "Unknown"
        predictions.append(name)
        distances.append(avg_distance)
    dirs.append(img_path)
    print("\nDistances:", closest_distances, "\nName:", name)

base_data['predicted'] = predictions
base_data['dir'] = dirs    
base_data['distance'] = distances     
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
print("Average Distance:", sum([distance for distance in distances if distance<0.5])/len([distance for distance in distances if distance<0.5]))
# Rachel Green, Chandler Bing, Elon Musk