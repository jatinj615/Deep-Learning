#Artificial Neural Network

#Installing Theano

#Installing Tensorflow

#Install Keras

#Inporting Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Split into train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Make The ANN

#import Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialise ANN
clf = Sequential()

# Adding the input layer and the first hidden layer
clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu', input_dim = 11))
clf.add(Dropout(rate=0.1))

# Adding Second Hidden layer
clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu'))
clf.add(Dropout(rate=0.1))

# Adding Output layer
clf.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation='sigmoid')) #use suftmax function in more than two categories

# Compiling ANN
clf.compile(optimizer= 'adam', loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'

# Fitting ANN to Training Set
clf.fit(X_train, y_train, batch_size=10, epochs=100)

#predict the test set result
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion Metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
"""Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000 """

new_pred = clf.predict(sc.fit_transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)


# Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_clf():
    clf = Sequential()
    clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu', input_dim = 11))
    clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu'))
    clf.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation='sigmoid')) #use suftmax function in more than two categories
    clf.compile(optimizer= 'adam', loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'
    return clf

clf = KerasClassifier(build_fn= build_clf, batch_size = 10, epochs = 100)

scores = cross_val_score(estimator= clf, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = np.mean(scores)
variance = np.std(scores)


#Tuning the ANN
##Using Grid Search
from sklearn.model_selection import GridSearchCV

def build_clf(optimizer):
    clf = Sequential()
    clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu', input_dim = 11))
    clf.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu'))
    clf.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation='sigmoid')) #use suftmax function in more than two categories
    clf.compile(optimizer= optimizer, loss='binary_crossentropy', metrics = ['accuracy']) # In case of more than two categories loss function equals 'categorical_crossentropy'
    return clf

clf = KerasClassifier(build_fn= build_clf)
params = {'batch_size': [25, 32],
          'epochs': [100, 500],
          'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = clf, param_grid = params, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_




