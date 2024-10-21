# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:53:55 2019

@author: Ava Fgh
"""

import glob
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D, Input , Flatten , Dense
from keras.optimizers import Adam
from keras import models,losses,layers,optimizers


# Build the model
data = []
for i in range(1,224):
    
    path = "C:/Users/project/images/{}/".format(i)
    images_paths = glob.glob(path +"*.png") + glob.glob(path +"*.jpg")

    info_path = glob.glob(path +"*.txt")
    info_file = open(info_path[0],'r');
    
    age =    int(float(info_file.readline().split(":")[1].replace("\n","").strip()))
    print ("{} --> {} ".format(i,age))
    
    for img in images_paths:
        image = cv2.imread(img)
        image = cv2.resize(image,(64,64));
        image = image / np.max(image)
        image = image.astype(np.float32)
        
        content = {"image" : image,"age" : age}
        data.append(content)
        
xTrainImage =[]
yTrainAge = []
for i in data:
    xTrainImage.append(i['image'])
    yTrainAge.append(i['age'])
    
yTrainAge = np_utils.to_categorical(yTrainAge)   
xTrainImage =np.array(xTrainImage)
print(xTrainImage.shape)

model=models.Sequential()
model.add(layers.Conv2D(250,2,activation='relu',padding='same',input_shape=(64,64,3)))
model.add(layers.MaxPool2D(pool_size = 2))
model.add(layers.Conv2D(128,2,activation='relu',padding='same'))
model.add(layers.MaxPool2D(pool_size = 2))
model.add(layers.Conv2D(83,(2,2),activation='relu',padding='same'))
model.add(layers.MaxPool2D(pool_size = 2))
model.add(layers.Flatten())
model.add(layers.Dense(83,activation = 'softmax'))

model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss = losses.categorical_crossentropy, metrics=['accuracy'])

temp=model.fit(xTrainImage, yTrainAge,validation_split=0.1 ,epochs=2,batch_size=128)
result=temp.history

model.save('model.h5')




# Test the model
'''
predict=[]
ppath="C:/Users/Desktop/20191129_160317.jpg"
pimg = cv2.imread(ppath)
pimg = cv2.resize(pimg,(64,64));
pimg = pimg / np.max(pimg)
pimg = pimg.astype(np.float64)
predict.append(pimg)

ppath="C:/Users/Desktop/2.jpg"
pimg = cv2.imread(ppath)
pimg = cv2.resize(pimg,(64,64));
pimg = pimg / np.max(pimg)
pimg = pimg.astype(np.float64)
predict.append(pimg)

ppath="C:/Users/Desktop/3.jpg"
pimg = cv2.imread(ppath)
pimg = cv2.resize(pimg,(64,64));
pimg = pimg / np.max(pimg)
pimg = pimg.astype(np.float64)
predict.append(pimg)


predict = np.array(predict)
tt = model.predict(predict)
tt= np.argmax(tt,axis=1)
print(tt)
'''