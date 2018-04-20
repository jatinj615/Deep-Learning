# -*- coding: utf-8 -*-
# Building CNN
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
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
with tf.device('/gpu:0'):
    clf.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
    

from keras.models import model_from_json
model_json = clf.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
clf.save_weights("model.h5")

#load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'


# making single predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis= 0)
result = clf.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'