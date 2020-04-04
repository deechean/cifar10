#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:23:43 2019

@author: Deechean
"""

import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import time 
from PIL import Image
#import cv2

def load(file_name):
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        return data  

class cifar10(object): 
    def __init__(self,size = 32, path='cifar-10-batches-py/', ramdom_wrap=True, random_flip=True, random_distort=True):        
        self.data_path = path
        self.train_images, self.train_labels = self._get_train()
        self.test_images, self.test_labels = self._get_test()
        self.image_size = size
        self.wrap = ramdom_wrap
        self.flip = random_flip
        self.distort = random_distort
        self.label_dic = {0:'aircraft', 1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        self._get_shuffle_index()

    def _get_shuffle_index(self):
        self.train_index = 0        
        self.test_index = 0        
        self.train_list_index = list(range(len(self.train_labels)))
        random.shuffle(self.train_list_index)
        #print(self.train_list_index)        
        

    def _get_train(self):
        train_labels = []        
        data1 = load(self.data_path+'data_batch_1')
        x1 = np.array(data1[b'data'])
        y1 = data1[b'labels']
        train_data = np.array(x1)
        train_labels = np.array(y1)
        
        data2 = load(self.data_path+'data_batch_2')
        x2 = np.array(data2[b'data'])
        y2 = data2[b'labels']
        train_data = np.append(train_data, x2)
        train_labels = np.append(train_labels, y2)

        data3 = load(self.data_path+'data_batch_3')
        x3 = np.array(data3[b'data'])
        y3 = np.array(data3[b'labels']).reshape(10000)
        train_data = np.append(train_data, x3)
        train_labels = np.append(train_labels, y3)

        data4 = load(self.data_path+'data_batch_4')
        x4 = np.array(data4[b'data'])
        y4 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x4)
        train_labels = np.append(train_labels, y4)
        
        data5 = load(self.data_path+'data_batch_5')
        x5 = np.array(data4[b'data'])
        y5 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x5)
        train_labels = np.append(train_labels, y5)
        
        train_data = train_data.reshape(-1, 3, 32, 32)
        train_labels.astype(np.int64)
        
        #for item in labels:
        #    train_labels.append(item)
        #print('image shape:',np.shape(train_data))
        #print('label shape:',np.shape(train_labels))  
        if len(train_data) != len(train_labels):
            assert('train images ' + str(len(train_data))+' doesnt equal to train labels' + str(len(train_labels)))
            
        print('train set length: '+str(len(train_data)))
        return train_data, train_labels
 
    def _get_test(self):
        test_labels = list()
        data1 = load(self.data_path+'test_batch')
        x = np.array(data1[b'data']).reshape(-1, 3, 32, 32)
        y = data1[b'labels']
        
        for item in y:
            test_labels.append(item)
        #print('test image shape:',np.shape(x))
        #print('test label shape:',np.shape(test_labels))        
        print('test set length: '+str(len(x)))
        return x, test_labels
    
    def _resize(self,image):
        resized_image = np.ndarray.reshape(image,(32,32,3))[2:30,2:30,0:3] 
        #print(resized_image.shape)
        return resized_image
    
    def random_flip(self,image):
        if random.random() < 0.5:
            image = np.flipud(image)
        return image.astype('uint8')
    
    def random_distort(self,image):
        if random.random() < 0.5:
            x = random.randint(1,10)
            y = random.randint(1,10)
            image = self.resize_image(image, (self.image_size+x,self.image_size+y))
            image = image[:, :self.image_size, :self.image_size]
        return image.astype('uint8')
    
    def random_bright(self, image, delta=32):      
        if random.random() < 0.3:   
            image = image/255.0
            image = np.power(image,0.5)*255
        return image.astype('uint8')
    
    def resize_image(self, image, new_size):
        img = Image.fromarray(image.transpose(1,2,0))       
        return np.asarray(img.resize(new_size,resample=Image.BILINEAR)).transpose(2,0,1)
    
    def ramdom_wrap(self, image):        
        if random.random() < 0.5:
            max_offsetx = random.randint(1,8)
            max_offsety = random.randint(1,8)
            image = image[:,max_offsetx:-max_offsetx,max_offsety:-max_offsety]
            image = self.resize_image(image, (self.image_size,self.image_size))
        return image.astype('uint8')
    
    def get_train_batch(self,batch_size=128):
        batch_image = []
        batch_label = []
        data_index = []
        i = 0
        for i in range(batch_size):
            d = self.train_images[self.train_list_index[self.train_index]]

            if self.wrap == True:
                d = self.ramdom_wrap(d)

            if self.flip == True:
                d = self.random_flip(d) 

            if self.distort == True:
                d = self.random_distort(d) 

            if self.image_size != 32:
                d = self.resize_image(d, (self.image_size, self.image_size))

            batch_image.append(d)
            batch_label.append(self.train_labels[self.train_list_index[self.train_index]])
            data_index.append(self.train_list_index[self.train_index])
            self.train_index += 1
            if self.train_index >=  len(self.train_images):
                self._get_shuffle_index()
                self.train_index = 0
        return np.array(batch_image).transpose(0,3,2,1).reshape(-1,self.image_size,self.image_size,3), batch_label, data_index
        
    def get_test_batch(self,batch_size=10000):
        batch_image = []
        batch_label = []
        data_index = []
        i = 0
        for i in range(batch_size):                
            d = self.test_images[self.test_index]
            if self.image_size != 32:
                d = self.resize_image(d, (self.image_size, self.image_size))                    
            batch_image.append(d) 
            batch_label.append(self.test_labels[self.test_index])            
            data_index.append(self.test_index)
            self.test_index += 1
            if self.test_index >=  len(self.test_images):
                self.test_index = 0
        return  np.array(batch_image).transpose(0,3,2,1).reshape(-1,self.image_size,self.image_size,3), batch_label, data_index
    
    