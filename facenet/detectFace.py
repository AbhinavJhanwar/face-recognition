# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:39:51 2019

@author: abhinav.jhanwar
"""

import cv2
# mathematical tools
import numpy as np
# time modules
from time import time
# models saving modules
import pickle
# image/video processing modules
from imutils.video import FPS
import imutils
import tensorflow as tf
from tensorflow.python.platform import gfile
import os, re
import glob
from tqdm import tqdm

base_dir = ""
model_cfg = "yolov3-face.cfg"
model_weights = "yolov3-wider_16000.weights"
nms_threshold = 0.3
yolo_conf_threshold = 0.7
svm_threshold = 0.4
img_size = 160
margin = 32
path = "Abhinav.jpg"
base_validate = "testing"


# define deep neural network parameters
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file  

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image
 
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in tqdm(range(nrof_samples)):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (image_size, image_size))
        if do_prewhiten:
            img = prewhiten(img)
        #img = crop(img, do_random_crop, image_size)
        #img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

paths=[]
for img_path in tqdm(glob.glob(os.path.join(base_validate, "*.jpg"))):  
    paths.append(img_path)
        
with tf.Graph().as_default():
    with tf.Session() as sess:
        load_model('20180402-114759.pb')
            
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
           
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        emb_array = np.zeros((nrof_images, embedding_size))
        
        images = load_data(paths, False, False, img_size)
        feed_dict = {images_placeholder:images, phase_train_placeholder:False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)

        # load classifier
        pickle_in = open("svm_clf.pickle","rb")
        classifier = pickle.load(pickle_in)
        pickle_in.close()
        
        classifier.predict_proba(emb_array)
        classifier.classes_
        