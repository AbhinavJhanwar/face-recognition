# face-recognition
various open source implementations for face detection and classification

##################################################################################
######################### FACE RECOGNITION #######################################
##################################################################################
Face recognition findings-
	1) Face Detection-
		a. Tested detection models - dlib (hog, cnn), resnet(performs best on speed), facenet and yolo (performs best in accuracy)
			i. Hog - performance in accuracy is very poor - discarded
			ii. Cnn - performance in accuracy is good but speed on cpu is very poor- discarded
			iii. Resnet - performance in accuracy is good but not with side faces or in dark - discarded
			iv. Yolo - performance in accuracy is very good but speed is ok - accepted as still best one yet faced
		b. Limitations of yolo - 
			i. Rolling angle is limited after a particular angle (approx 30) doesn't detect faces properly
			Remedy- faces are kept in <30 rolling angle
			ii. On side faces sometimes crops the face from eye portion in left/right 
			Remedy- added margin in both left and right side, also instead of fixed value we are taking percent of width of boundary box
			iii. Boundary box of face is not necessarily inside the image dimensions so have to take care of this part 
			Remedy- applied condition-
				1) For left- max(left, 0)
				2) For right- min(right, frame_width)
				3) For top- max(top, 0)
				4) For bottom- min(bottom, frame_height)
			iv. For side faces boundary box sometimes crops the facial image
			Remedy- extended boundary box dimensions in left and right side by 10% of boundary box dimensions
	2) Face Encoding/Embedding (face_recognition-dlib)-
		a. Using dlib based 128 dimensional vectors
		b. Limitations- works only for front face as assumes two eyes in each image and hence side face encoding may not be performing well (pitch angle: 45 to -45)
		Remedy- trying to filter out only front face images using face pose detection
	3) Face Classification Model-
		a. Algorithms tested - svm, knn
			i. Svm- no control on how to decide a face is known or unknown other than probability- overall performance was not good
			ii. Knn- classification can be controlled in more intense way in terms of probability (capability to control the number of neighbors) and distance (comparing face encoding vector distances)
		b. Face data augmentation to be trained-
			i. Filtered Front faces 
			ii. Varying lighting conditions 
			iii. Image magnification
			iv. Horizontal flipping
			v. Stretching image in width and height
		c. Limitations - 
			i. Unknown faces side poses gives very low distance from known faces and hence miss classified
			Remedy- Only front faces of images are trained and separate thresholds and accuracies for front and side faces
			ii. Sometimes front pose of unknown faces also gives low distances (for different faces average distance is varying. Example for Abhinav average distance was 0.31 while for Sandeep's face it was 0.28)
			Remedy- Dynamic threshold based on distinct faces i.e. for Abhinav threshold is 0.31+margin and for Sandeep threshold is 0.28+margin (margin is a gap decided based on testing on known/unknown faces to extend the threshold)
			iii. Yaw angle is limited 15 to -10
			iv. Pitch angel is limited: 45 to -45
		
		d. Other Remedies-
			i. Histogram equalization of face before encoding
			ii. In detection to remove garbage faces- filtering images with resnet so that only good quality faces are encountered and then yolo detects the actual face and encoding is performed)


##################################################################################
################## Training face detection #######################################
##################################################################################

1) Clone repository
2) Create folder data
3) Add folders with images of persons to be trained inside data folder as following-
	a) Suppose persons to be trained are 'David' and 'Smith'
	b) Create two folders with names 'David' and 'Smith'
	c) Add all images of 'David' in folder 'David' and all images of 'Smith' in folder 'Smith'
2) open 'config_training.json' file and update the following configuration-
	a) "user_images" : "data/"
3) open anaconda command prompt
4) run command: 'python Real_Time_Face_Recognition_OOBS.py'


##################################################################################
################## Start face detection ##########################################
##################################################################################
1) open anaconda command prompt
2) run command: 'python Real_Time_Face_Recognition_OOBS.py'

