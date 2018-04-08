# -*- coding: utf-8 -*-
# Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialising CNN
clf = Sequential()

# Step1 - convolution layer

clf.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3 ), activation= 'relu'))

