# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:34:45 2018

@author: abhinav.jhanwar
"""

''' hog + svm'''

import face_recognition
import cv2
from imutils.video import FPS
import pickle
from sklearn.svm import SVC
import os, glob
import numpy as np
from tqdm import tqdm

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
            for img_path in tqdm(glob.glob(os.path.join(base, "*"))):
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

    # Save the trained KNN classifier
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
video_capture = cv2.VideoCapture(0)

fps = FPS().start()
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if ret==False:
        print("frame not captured")
        continue

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_encoding = [face_encoding]
        probabilities = svm_clf.predict_proba(face_encoding)[0]
        confidence = max(probabilities)
        print(probabilities)
        
        name = "Unknown"
        if confidence>confidence_threshold:
            name = svm_clf.classes_[np.argmax(probabilities)]
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+15), font, 0.5, (255, 255, 255), 1)
        
        if name!='Unknown':
            # label for confidence
            cv2.rectangle(frame, (left, top-20), (right, top), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, str(round(confidence*100,2))+'%', (left + 6, top-3), font, 0.5, (255, 255, 255), 1)


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
video_capture.release()
cv2.destroyAllWindows()