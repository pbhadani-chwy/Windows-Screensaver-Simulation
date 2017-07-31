# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:22:01 2017

@author: pbhadani
"""

import os
from random import randint
from scipy import ndimage, misc
from skimage import io
from itertools import islice
from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten,Reshape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import time
import warnings
import numpy as np
from numpy import newaxis
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import pandas as pd
import re

label = []
i=0


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0:49152]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0:49152])
	return np.array(dataX), np.array(dataY)


images = np.ndarray(shape=(12,3,128,128),dtype=np.float32)
for root, dirnames, filenames in os.walk("C:\\Users\\pbhadani\\Documents\\ScreenSaver_LSTM\\code\\images"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        head,tail = os.path.split(filepath)  #split file into path + image name
        label.append(head.split('\\')[6])
        #print(label)
        image = load_img(filepath)
        r,c,ch=img_to_array(image).shape
        image=image.resize((128,128),Image.LANCZOS)
        image=img_to_array(image)
        images[i]=image.reshape(3,128,128)
        i += 1

fin_image = images
for i in range(9):
    
    fin_image = np.concatenate((fin_image,images),axis = 0)
    
test1 = fin_image[117: , : , : , :]
training_set = fin_image
training_set /= 255.0

testing_set = test1
testing_set /= 255.0

print(images)
for i in range (12):
    fig,ax = plt.subplots(1)
    
    # Display the image
    #index=randint(0,49)
    ax.imshow(trainY[i].reshape(128,128,3))  
    # Create a Rectangle patch
    #x=rx[index]-lx[index]
    #y=ry[index]-ly[index]
    #rect1 = patches.Rectangle((lx[i],ly[i]),rx[i],ry[i],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    #ax.add_patch(rect1)
    
    plt.show()

training_set = fin_images
print(training_set[42][1][1][1])
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

'''test_set = training_set.reshape(43,518400).reshape(43,518400,1)
train_set = np.roll(test_set,1,axis = 0).reshape(43,518400,1)'''

look_back = 1
trainX, trainY = create_dataset(training_set, look_back)

inputX,inputY = create_dataset(testing_set, look_back)

testX = inputX.reshape(1,1,49152)
testY = inputY.reshape(1,49152)

trainX = trainX.reshape(118,1,49152)
trainY = trainY.reshape(118,49152)

print(trainX.shape)
'''trainX = np.reshape(trainX, (trainX.shape[0] * 3072, 4, 1))
trainY = trainY.reshape((1,trainY.shape[0] * 3072))'''



# Feature Scaling
'''for i in range(43):
    for j in range(3):
        for k in range(360):
            for l in range(480):
                training_set[i][j][k][l] = training_set[i][j][k][l]/255'''

class MyReshape(Layer):
    def get_output(self, train):
        X = self.get_input(train)
        nshape = (1,) + X.shape 
        return theano.tensor.reshape(X, nshape)

# Feature scaling 
training_set = training_set/255.0

inp = Input(shape=(3,360,480))
flat = Flatten()(inp)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 50, activation = 'sigmoid', input_shape = (1,49152),dropout=0.2,recurrent_dropout=0.2))

# Adding the output layer
regressor.add(Dense(units = 49152,activation = 'sigmoid'))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the RNN to the Training set
regressor.fit(trainX, trainY, batch_size = 16, epochs = 1000,shuffle = False)


predicted_value = regressor.predict(testX)
predicted_value = predicted_value.reshape(1,3,128,128)
predicted_value *= 255.0

for i in range (2):
    fig,ax = plt.subplots(1)
    
    # Display the image
    #index=randint(0,49)
    ax.imshow(predicted_value[i].reshape(128,128,3))  
    # Create a Rectangle patch
    #x=rx[index]-lx[index]
    #y=ry[index]-ly[index]
    #rect1 = patches.Rectangle((lx[i],ly[i]),rx[i],ry[i],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    #ax.add_patch(rect1)
    
    #plt.show()


