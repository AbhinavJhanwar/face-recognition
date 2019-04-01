# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:08:45 2018

@author: abhinav.jhanwar
"""


import face_recognition
import cv2
import os
import glob
import time
import pickle
from imutils.video import FPS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# CNN MODEL LIBRARIES
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils


# load pre-traind encodings
pickle_in = open("SavedModels/known_face_encodings.pickle","rb")
known_face_encodings = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("SavedModels/known_face_names.pickle","rb")
known_face_names = pickle.load(pickle_in)
pickle_in.close()

# get name of person to be trained
names = input("Write names of persons to be trained:\n").split(',')

if len(names[0])>0:
    print("Encoding Images...\n")
    for name in names:
        base = "known_people"
        name = name.strip()
        # find images of person to be trained
        base = os.path.join(base, name)
        for path in glob.glob(os.path.join(base, "*")):
            print(path)
            image = face_recognition.load_image_file(path)
            try:
                # using cnn to train models
                face_locations = face_recognition.face_locations(image, model="cnn")
                image_encoding = face_recognition.face_encodings(image, face_locations, num_jitters=5)[0]
                known_face_encodings.append(image_encoding)
                known_face_names.append(name)    
            except:
                print("No face Detected in ",path)
    # save the trained models
    pickle_out = open("SavedModels/known_face_names.pickle","wb")
    pickle.dump(known_face_names, pickle_out)
    pickle_out.close()
    
    pickle_out = open("SavedModels/known_face_encodings.pickle","wb")
    pickle.dump(known_face_encodings, pickle_out)
    pickle_out.close()
    
    print("Encoding Completed!\n")

else:
    print("Encoding Skipped!\n")
    
    
X_train = pd.DataFrame(known_face_encodings)

# prepare target
le = LabelEncoder()
y_train = le.fit_transform(known_face_names)
# save label encoding
pickle_out = open("SavedModels/y_train_le.pickle","wb")
pickle.dump(le, pickle_out)
pickle_out.close()

y_encoded = np_utils.to_categorical(y_train)


# DEFINE CNN
def build_classifier(optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 128, activation = 'relu', input_dim = 128))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = len(le.classes_), activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def get_Parameters():
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [25, 100],#[25, 32],
                  'epochs': [100, 200],
                  'optimizer': ['adam', 'rmsprop']}
    
    # scoring: neg_mean_squared_error for regression
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(best_parameters, best_accuracy)

#get_Parameters()

def trainClassifier():
    classifier = Sequential()
    classifier.add(Dense(units = 128, activation = 'relu', input_dim = 128))
    classifier.add(Dense(units = 300, activation = 'relu'))
    classifier.add(Dense(units = len(le.classes_), activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #classifier = load_model("SavedModels/model.h5")
    history = classifier.fit(X_train, y_encoded, batch_size = 25, epochs = 100)
    accuracy = history.history['acc']
    loss = history.history['loss']
    
    if not os.path.exists("SavedModels"):
                os.makedirs("SavedModels")
    classifier.save("SavedModels/model.h5")
    return classifier

classifier = trainClassifier()

######## testing ###################
test_image = face_recognition.load_image_file("Abhinav.jpg")
location = face_recognition.face_locations(test_image)
test_image_encoding = face_recognition.face_encodings(test_image, location)[0]
if type(test_image_encoding)!='list':
    test_image_encoding = [test_image_encoding]
    
classifier = load_model("SavedModels/model.h5")
prediction = list(classifier.predict(pd.DataFrame(test_image_encoding))[0])
confidence = max(prediction)
name = le.classes_[np.argmax(prediction)]

