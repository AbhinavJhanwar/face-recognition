# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:13:33 2018

@author: abhinav.jhanwar
"""

''' recognize face and draw box '''

import face_recognition
from PIL import Image, ImageDraw, ImageFont
import time

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
Abhinav_image = face_recognition.load_image_file("known_people/Abhinav.jpg")
Abhinav_image_encoding = face_recognition.face_encodings(Abhinav_image)[0]


# Load a thired sample picture and learn how to recognize it.
Jothi_image = face_recognition.load_image_file("known_people/Jothi.jpg")
Jothi_face_encoding = face_recognition.face_encodings(Jothi_image)[0]

# Create arrays of known face encodings and their names
known_face_encoding = [
    Abhinav_image_encoding,
    Jothi_face_encoding
    ]

known_face_names = [
        "Abhinav Jhanwar",
        "Jothi"
        ]

# Load an image with an unknown face/faces
#unknown_image = face_recognition.load_image_file("unknown_people/Unknown.jpg")
#unknown_image = face_recognition.load_image_file("groups/business-people-1.jpg")
#unknown_image = face_recognition.load_image_file("groups/3809.jpg")
unknown_image = face_recognition.load_image_file('outside_000001.jpg')

t1 = time.time()
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image, model="hog")
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)

# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    font = ImageFont.truetype('arial.ttf',30)
    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom), (right, bottom+text_height+20)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom), name, fill=(255, 255, 255, 255), font=font)


# Remove the drawing library from memory as per the Pillow docs
del draw

t2 = time.time()

# Display the resulting image
pil_image.show()
print("Total time taken in seconds: ",t2-t1)

# You can also save a copy of the new image to disk if you want by uncommenting this line
# pil_image.save("image_with_boxes.jpg")











