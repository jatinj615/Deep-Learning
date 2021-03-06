# -*- coding: utf-8 -*-

# Part 1 Identify the Frauds with Self-Organizing Map
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Training SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualizing the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5, 1)], mappings[(1, 5)]), axis = 0)
frauds = scaler.inverse_transform(frauds)

# Going from unsupervised to supervised

# create matrix of features

customers = dataset.iloc[:, 1:].values

# creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# fitting to ANN

# scaling features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Make The ANN

#import Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialise ANN
clf = Sequential()

# Adding the input layer and the first hidden layer
clf.add(Dense(output_dim = 2, kernel_initializer = 'uniform', activation='relu', input_dim = 15))
clf.add(Dropout(rate=0.1))

# Adding Output layer
clf.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation='sigmoid')) #use suftmax function in more than two categories

# Compiling ANN
clf.compile(optimizer= 'adam', loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'

# Fitting ANN to Training Set
clf.fit(customers, is_fraud, batch_size=1, epochs=3)

y_pred = clf.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
