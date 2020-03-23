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
    def __init__(self,size = 32, path='cifar-10-batches-py/'):
        self.train_indexs = list()
        self.test_indexs = list()
        self.data_path = path
        self.train_images, self.train_labels = self._get_train()
        self.test_images, self.test_labels = self._get_test()
        self.image_size = size
        self.label_dic = {0:'aircraft', 1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

        
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
    
    def random_flipper(self,image):
        if random.random() < 0.5:
            image = np.flipud(image)
        return image
    
    # 预处理函数
    # im中的像素值为 [0, 255] 闭区间， 则 (1+im) 为 [1, 256]
    # 先做 (1+im)/257 操作将值归一化到 (0, 1) 开区间内
    # 再使用 sigmoid函数 的反函数，效果见sigmod函数图像
    # -np.log(1/((1 + 0)/257) - 1) = -5.5451774444795623
    # -np.log(1/((1 + 255)/257) - 1) = 5.5451774444795623
    def do_normalise(self,im):
        return -np.log(1/((1 + im)/257) - 1)  
    
    # 预处理函数的反函数
    # 即先使用sigmod函数，再将值变换到(0, 257)区间再减1，通过astype保证值位于[0, 255]
    # 关于 astype("uint8") ：
    # np.array([-1]).astype("uint8") = array([255], dtype=uint8)
    # np.array([256]).astype("uint8") = array([0], dtype=uint8)
    def undo_normalise(self,im):
        return (1/(np.exp(-im) + 1) * 257 - 1).astype("uint8")
    
    def rotation_matrix(self,theta):
        """
        3D 旋转矩阵，围绕X轴旋转theta角
        """
        return np.c_[
            [1,0,0],
            [0,np.cos(theta),-np.sin(theta)],
            [0,np.sin(theta),np.cos(theta)]
            ]
    
    
    def distort_color(self,image):
        if random.random() < 0.5:
            image = self.do_normalise(image)
            image = image.transpose((1,2,0))
            image = np.einsum('ijk,lk->ijl', image, self.rotation_matrix(np.pi/4))
            # 利用爱因斯坦求和约定做矩阵乘法，实际上是将每个RGB像素点表示的三维空间点绕X轴（即红色通道轴）旋转180°。
            image = image.transpose((2,0,1))
            image = self.undo_normalise(image)            
        return image
        
    
    def random_bright(self, image, delta=32):      
        if random.random() < 0.3:   
            image = image/255.0
            image = np.power(image,0.5)*255
        return image
   
    def get_train_batch(self,batch_size=128, augument = True):
        batch_image = list()
        batch_label = list()
        data_index = list()
        i = 0
        while i < batch_size:
            index = random.randint(0, len(self.train_labels)-1)
            if not index in self.train_indexs:
                i += 1
                d = self.train_images[index]
                if augument:
                    #d = self.ramdom_wrap(self.random_bright(self.random_flipper(d)))
                    #d = self.random_flipper(d)
                    d = self.ramdom_wrap(self.random_flipper(d))
                    d = d.astype('uint8')
                    
                batch_image.append(d)
                batch_label.append(self.train_labels[index])
                self.train_indexs.append(index)
                data_index.append(index)
                if len(self.train_indexs) >=  len(self.train_images):
                    self.train_indexs.clear()
        return np.array(batch_image).transpose(0,3,2,1).reshape(-1,32,32,3), batch_label, data_index
        
    def get_test_batch(self,batch_size=10000):
        batch_image = list()
        batch_label = list()
        data_index = list()
        i = 0
        while i < batch_size:
            index = random.randint(0, len(self.test_labels)-1)
            if not index in self.test_indexs:
                i += 1
                d = self.test_images[index]
                batch_image.append(d) 
                batch_label.append(self.test_labels[index])
                self.test_indexs.append(index)
                data_index.append(index)
                if len(self.test_indexs) >=  len(self.test_images):
                    self.test_indexs.clear()
        return  np.array(batch_image).transpose(0,3,2,1).reshape(-1,32,32,3), batch_label,data_index    
    
    def resize_image(self, image, new_size):
        img = Image.fromarray(image.transpose(1,2,0))       
        return np.asarray(img.resize(new_size,resample=Image.BILINEAR)).transpose(2,0,1)
    
    def ramdom_wrap(self, image):        
        if random.random() < 0.5:
            max_offsetx = random.randint(1,8)
            max_offsety = random.randint(1,8)
            image = image[:,max_offsetx:-max_offsetx,max_offsety:-max_offsety]
            image = self.resize_image(image, (self.image_size,self.image_size))
        return image