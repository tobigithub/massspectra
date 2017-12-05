#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:16:37 2017

@author: josharnold
"""

### currently best:
### Test accuracy: 0.681318685249
### using normalizer
### Test accuracy: 0.703296705262
### using batch size =20 and  Normalizer()   and  MaxAbsScaler()
### Test accuracy: 0.747252751183
    

import pandas as pd

# required for accuracy reporting reproducibility
import numpy as np
np.random.seed(12345)

from keras.models import Sequential 
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Part 1: import data

# Import data set
data_filename = 'data.csv'
data = pd.read_csv(data_filename, sep=',', decimal='.', header=None)
y = data.loc[1:, 1:400].values
X = data.loc[1:, 401:1591].values

print (y.shape)
print (X.shape)

# Split data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# scaler = MinMaxScaler()    # Test accuracy: 0.670329687
scaler = Normalizer()       # Test accuracy: 0.714285716251
scaler = MaxAbsScaler()     # Test accuracy: 0.725274729205
#scaler = StandardScaler()  # Test accuracy: 0.571428572739
#scaler = RobustScaler()    # Test accuracy: 0.56043956175
#scaler = Normalizer() and 
#scaler = MaxAbsScaler()    #Test accuracy: 0.736263740194  

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Part 2: create ANN and fit data
def baseline_model():
    # Intialize the artificial neural network
    model = Sequential()

    # Input layer and hidden layer 
    model.add(Dense(activation="relu", input_dim=1191, units=700, kernel_initializer="glorot_normal"))

    # Dropout to aid overfitting
    model.add(Dropout(0.285))


    # Output layer
    model.add(Dense(activation="relu", input_dim=700, units=400, kernel_initializer="uniform"))

    # Compile the ANN
    model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=["accuracy","mean_squared_error"])
    
    return model

# Fit the ANN to the training set
model = baseline_model()
result = model.fit(X_train, y_train, batch_size=20, nb_epoch=1600, validation_data=(X_test, y_test))


# Part 3: analyze results

# summarize history for accuracy
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Print final loss and accuracy 
score = model.evaluate(X_test, y_test)
print("")
print("")
print("")
print("******************************")
print('Test score:', score[0]) 
print('Test accuracy:', score[1])
print("******************************")
print("")
print("")
print("")

