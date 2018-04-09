# -*- coding: utf-8 -*-
# Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialising CNN
clf = Sequential()

# Step1 - convolution layer

clf.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3 ), activation= 'relu'))

# Step2 - Max Pooling
clf.add(MaxPooling2D(pool_size= (2, 2)))

#Adding another convolutional layer
clf.add(Convolution2D(32, (3, 3), activation= 'relu'))
clf.add(MaxPooling2D(pool_size= (2, 2)))

#Step3 - Flattening
clf.add(Flatten())

# Step4 - Full connection
clf.add(Dense(output_dim = 128, activation = 'relu'))
clf.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Step5 compile CNN
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

clf.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)


