# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:17:48 2018

@author: abhinav.jhanwar
"""

''' Recognize faces in images and identify who they are '''

import face_recognition

picture_of_me = face_recognition.load_image_file("Abhinav.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("Unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
    

################ try with multiple images #####################
# Load the jpg files into numpy arrays
Abhinav_image = face_recognition.load_image_file("known_people/Abhinav.jpg")
jobs_image = face_recognition.load_image_file("known_people/steve-jobs.jpg")
unknown_image = face_recognition.load_image_file("unknown_people/Unknown.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    Abhinav_face_encoding = face_recognition.face_encodings(Abhinav_image)[0]
    jobs_face_encoding = face_recognition.face_encodings(jobs_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    Abhinav_face_encoding,
    jobs_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Abhinav? {}".format(results[0]))
print("Is the unknown face a picture of jobs? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
    
    
    
