# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:19:39 2019

@author: abhinav.jhanwar
"""
import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import json
import shutil
from distutils.util import strtobool
from tqdm import tqdm

class imageDataAugmentation:
    __slots__ = "datagen"
    def __init__(self, config):        
        # this is the augmentation configuration we will use for training
        self.datagen = ImageDataGenerator(
                rotation_range=config['rotation_range'],
                #width_shift_range=config['width_shift_range'],
                #height_shift_range=config['height_shift_range'],
                brightness_range=config['brightness_range'],
                shear_range=config['shear_range'],
                zoom_range=config['zoom_range'],
                channel_shift_range=config['channel_shift_range'],
                fill_mode=config['fill_mode'],
                horizontal_flip=strtobool(config['horizontal_flip']),
                vertical_flip=strtobool(config['vertical_flip']),
                rescale=config['rescale']
                )
    def generateData(self, config):
        # fetch all the folder names
        names = []
        dirs = os.listdir(config['directory'])
        for name in dirs:
            if name not in os.listdir(config['save_to_dir']):
                names.append(name)
        
        print('[INFO] generating data for following images -\n', names)
        
        for name in tqdm(names):
            # read first image for fetching the dimensions
            img = load_img(os.path.join(config['directory'], name, os.listdir(os.path.join(config['directory'], name))[0]))
            
            # remove any existing directories for the given name
            if os.path.exists(os.path.join(config['save_to_dir'], name)):
                #shutil.rmtree(os.path.join(config["save_to_dir"],name))
                continue
                
            # save the original images
            shutil.copytree(os.path.join(config['directory'], name), os.path.join(config['save_to_dir'], name))
        
            # this is a generator that will read pictures and generate batches of augmented image data
            generator = self.datagen.flow_from_directory(
                    directory = config['directory'],  # this is the image folders directory
                    target_size=(img.height, img.width),  # all images will be resized to size of actual images
                    color_mode=config["color_mode"],
                    class_mode=config["class_mode"],
                    batch_size=len(os.listdir(os.path.join(config['directory'], name))),
                    shuffle=strtobool(config["shuffle"]),
                    save_to_dir=os.path.join(config["save_to_dir"],name),
                    save_format=config["save_format"],
                    classes=[name]
                    )
                
            for i in tqdm(range(config['augmentBy'])):
                generator = self.datagen.flow_from_directory(
                        directory = config['save_to_dir'],  # this is the image folders directory
                        target_size=(img.height, img.width),  # all images will be resized to size of actual images
                        color_mode=config["color_mode"],
                        class_mode=config["class_mode"],
                        batch_size=len(os.listdir(os.path.join(config['save_to_dir'], name))),
                        shuffle=strtobool(config["shuffle"]),
                        save_to_dir=os.path.join(config["save_to_dir"],name),
                        save_format=config["save_format"],
                        classes=[name]
                        )
                next(generator)
               
if __name__=="__main__":
    with open('config.json', 'r') as config:
        config = json.load(config)
    data = imageDataAugmentation(config)
    data.generateData(config)
