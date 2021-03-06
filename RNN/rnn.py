# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importin training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = np.array(training_set.iloc[:, 1:2].values)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(training_set)
training_set = scaler.transform(training_set)

# inputs and outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1))
X_train

# Importing Libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Initialising RNN
regressor = Sequential()

# Adding input layer and LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# Adding output layer 
regressor.add(Dense(units = 1))

# Compiling RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN
regressor.fit(X_train, y_train, batch_size=32, epochs = 300)

#saving model

from keras.models import model_from_json
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")


# Visualising 
# Test Data
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = np.array(test_set.iloc[:, 1:2].values)
# scaling test set
test_set = scaler.transform(real_stock_price)

X_test = np.reshape(test_set, (20, 1, 1))

# predict the stock price
y_pred = regressor.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)

# Visualize
plt.plot(real_stock_price, color='red', label= 'Real Google Stock Price')
plt.plot(y_pred, color='blue' , label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# getting real google data
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = np.array(real_stock_price_train.iloc[:, 1:2].values)

# predict stock price
predicted = regressor.predict(X_train)

predicted = scaler.inverse_transform(predicted)

# Visualize
plt.figure(figsize=(10,8))
plt.plot(real_stock_price_train, color='red', label= 'Real Google Stock Price')
plt.plot(predicted, color='blue' , label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()

# Evaluating RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, y_pred))
percent = rmse/800
