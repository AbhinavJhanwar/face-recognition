# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:54:02 2018

@author: abhinav.jhanwar
"""

''' hog + svm'''

import face_recognition
import cv2
import pickle
from sklearn.svm import SVC
import os, glob
import numpy as np
import time

base_dir = "SavedModels/"

# load pre-traind encodings
pickle_in = open(base_dir+"known_face_encodings.pickle","rb")
known_face_encodings = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(base_dir+"known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)
pickle_in.close()

def train(model_save_path=None, verbose=False):
    # get name of person to be trained
    names = input("Write names of persons to be trained:\n").split(',')
    
    if len(names[0])>0:
        print("Encoding Images...\n")
        for name in names:
            base = "known_people"
            name = name.strip()
            # find images of person to be trained
            base = os.path.join(base, name)
            for img_path in glob.glob(os.path.join(base, "*")):
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image, model="cnn")
                if len(face_locations) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    known_face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=5)[0])
                    known_face_names.append(name)
                    
        # save the trained models
        pickle_out = open(base_dir+"known_face_names.pickle","wb")
        pickle.dump(known_face_names, pickle_out)
        pickle_out.close()
        
        pickle_out = open(base_dir+"known_face_encodings.pickle","wb")
        pickle.dump(known_face_encodings, pickle_out)
        pickle_out.close()
        
        print("Encoding Completed!\n")
    
    else:
        print("Encoding Skipped!\n")
    
  
    # Create and train the SVM classifier
    svm_clf = SVC(kernel='linear', random_state=42, probability=True)
    svm_clf.fit(known_face_encodings, known_face_names)

    # Save the trained SVM classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(svm_clf, f)

    return svm_clf


# STEP 1: Train the KNN classifier and save it to disk
print("Training SVM classifier...")
classifier = train(
                model_save_path="svm.clf",
                   )
print("Training complete!")

confidence_threshold = 0.5

#classifier = pickle.load("knn_model.clf")

# Get a reference to webcam #0 (the default one)
base = "Abhinav/"
name = base+"Abhinav6.jpg"
image = cv2.imread(name)

data_total_inference_time=0
start_time = time.time()

face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Loop through each face in this frame of video
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    face_encoding = [face_encoding]
    probabilities = classifier.predict_proba(face_encoding)[0]
    confidence = max(probabilities)
    print(confidence)
    
    name = "Unknown"
    if confidence>confidence_threshold:
        name = classifier.classes_[np.argmax(probabilities)]
    
    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    # Draw a label with a name below the face
    cv2.rectangle(image, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
    
    if name!='Unknown':
        # label for confidence
        cv2.rectangle(image, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, str(round(confidence*100,2))+'%', (left + 6, top-3), font, 0.5, (255, 255, 255), 1)

inf_time = time.time() - start_time
data_total_inference_time += inf_time 
print("[INFO] elapsed time: {:.2f}".format(data_total_inference_time))
cv2.imwrite("svm_detection.jpg",image)
 