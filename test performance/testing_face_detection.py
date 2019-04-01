# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:26:23 2018

@author: abhinav.jhanwar
"""

import face_recognition
import os
import numpy as np
from tqdm import tqdm
import cv2
import time

def extract_and_filter_data(splits):
    # Extract bounding box ground truth from dataset annotations, also obtain each image path
    # and maintain all information in one dictionary
    '''0--Parade/0_Parade_marchingband_1_849.jpg
    1
    449 330 122 149 0 0 0 0 0 0'''
    bb_gt_collection = dict()

    for split in splits:
        with open(
                os.path.join('wider_face_%s_bbx_gt.txt' % (split)), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.split('\n')[0]
            if line.endswith('.jpg'):
                image_path = os.path.join('images', line)
                bb_gt_collection[image_path] = []
            line_components = line.split(' ')
            if len(line_components) > 1:

                # Discard annotation with invalid image information, see dataset/wider_face_split/readme.txt for details
                if int(line_components[7]) != 1:
                    x1 = int(line_components[0])
                    y1 = int(line_components[1])
                    w = int(line_components[2])
                    h = int(line_components[3])

                    # In order to make benchmarking more valid, we discard faces with width or height less than 15 pixel,
                    # we decide that face less than 15 pixel will not informative enough to be detected
                    if w > 15 and h > 15:
                        bb_gt_collection[image_path].append(
                            np.array([x1, y1, x1 + w, y1 + h]))

    return bb_gt_collection

def evaluate(face_detector, bb_gt_collection, iou_threshold):
    total_data = len(bb_gt_collection.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0
    face_pred = []
    
    if face_detector == 'yolo':
        model_cfg = "yolov3-face.cfg"
        model_weights = "yolov3-wider_16000.weights"
        net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
     
    elif face_detector == 'opencv_resnet':
        modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
             
    # Evaluate face detector and iterate it over dataset
    for i, key in tqdm(enumerate(bb_gt_collection), total=total_data):
        #print(key)
        image_data = cv2.imread(key)
        face_bbs_gt = np.array(bb_gt_collection[key])
        total_gt_face = len(face_bbs_gt)

        start_time = time.time()
        if face_detector == 'dlib_hog':
            face_pred = np.array(face_recognition.face_locations(image_data, model="hog"))
            for i, value in enumerate(face_pred):
                face_pred[i][0], face_pred[i][1], face_pred[i][2], face_pred[i][3] = face_pred[i][3], face_pred[i][0], face_pred[i][1], face_pred[i][2]
                
        elif face_detector == 'dlib_cnn':
            face_pred = np.array(face_recognition.face_locations(image_data, model="cnn"))
            for i, value in enumerate(face_pred):
                face_pred[i][0], face_pred[i][1], face_pred[i][2], face_pred[i][3] = face_pred[i][3], face_pred[i][0], face_pred[i][1], face_pred[i][2]
               
        elif face_detector == 'yolo':
            blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (416, 416)), 1 / 255, (416, 416),
                                 [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layers_names = net.getLayerNames()
            outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
            (frame_height, frame_width) = image_data.shape[:2]
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        width = int(detection[2] * frame_width)
                        height = int(detection[3] * frame_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                       nms_threshold)
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                face_pred.append(np.array([left, top, left + width,
                             top + height]))
            print("\n", "total:",total_gt_face, "predicted:",len(indices))           
            
        elif face_detector == 'opencv_resnet':
            blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (300, 300)), 1.0,
	                                        (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            (h, w) = image_data.shape[:2]
            confidences = []
            boxes = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.3:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    # startX, startY, endX, endY
                    confidences.append(float(confidence))
                    boxes.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
            print(len(boxes))
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                       nms_threshold)
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                face_pred.append(np.array([left, top, left + width,
                             top + height]))
            print("\n", "total:",total_gt_face, "predicted:",len(face_pred))
            
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time

        ### Calculate average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            cv2.rectangle(image_data, (gt[0], gt[1]), (gt[2], gt[3]),
                          (255, 0, 0), 2)
            for i, pred in enumerate(face_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                cv2.rectangle(image_data, (pred[0], pred[1]),
                              (pred[2], pred[3]), (0, 0, 255), 2)
                iou = get_iou(gt, pred)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    if pred_dict[i] >= iou_threshold:
                        tp += 1
                precision = float(tp) / float(total_gt_face)
                
            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision
            

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

    result = dict()
    result['average_iou'] = float(data_total_iou) / float(total_data)
    result['mean_average_precision'] = float(data_total_precision) / float(
        total_data)
    result['average_inferencing_time'] = float(
        data_total_inference_time) / float(total_data)

    return result

def get_iou(boxA, boxB):
    """
	Calculate the Intersection over Union (IoU) of two bounding boxes.
	Parameters
	----------
	boxA = np.array( [ xmin,ymin,xmax,ymax ] )
	boxB = np.array( [ xmin,ymin,xmax,ymax ] )
	Returns
	-------
	float
		in [0, 1]
	"""

    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


splits = ['test']
iou_threshold = 0.5
# yolo
nms_threshold = 0.3
conf_threshold = 0.7

detector_list = [
        'opencv_resnet', 'dlib_hog', 'yolo', 'dlib_cnn'
    ]

face_detector = 'opencv_resnet'
data_dict = extract_and_filter_data(splits)
result = evaluate(face_detector, data_dict, iou_threshold)
print(face_detector)
print ('Average IOU = %s' % (str(result['average_iou'])))
print ('mAP = %s' % (str(result['mean_average_precision'])))
print ('Average inference time = %s' % (str(result['average_inferencing_time'])))
