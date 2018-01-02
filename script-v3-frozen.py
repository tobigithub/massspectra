#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:16:37 2017

@author: josharnold
"""

### Alkanes v3-frozen 
### Do not edit script !
### Epoch 04197: early stopping
### Test accuracy: 0.791208782039

### using normalizer
### Test accuracy: 0.703296705262

### using batch size =20 and  Normalizer()   and  MaxAbsScaler()
### Test accuracy: 0.747252751183

### using model.add(Dense(800, kernel_initializer='uniform', activation='relu'))
### Test accuracy: 0.78021977105

### Latest Test accuracy: 0.791208793174

import pandas as pd

# required for accuracy reporting reproducibility
import numpy as np
np.random.seed(12345)

from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, ActivityRegularization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

# Part 1: import data

# Import data set
data_filename = 'data.csv'
data = pd.read_csv(data_filename, sep=',', decimal='.', header=None)
y = data.loc[1:, 1:400].values
X = data.loc[1:, 401:1591].values

print ("original scale:")
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

# scaler = MinMaxScaler()   # Test accuracy: 0.670329687
scaler = Normalizer()       # Test accuracy: 0.714285716251
scaler = MaxAbsScaler()     # Test accuracy: 0.725274729205
#scaler = StandardScaler()  # Test accuracy: 0.571428572739
#scaler = RobustScaler()    # Test accuracy: 0.56043956175
#scaler = Normalizer() and 
#scaler = MaxAbsScaler()    #Test accuracy: 0.736263740194  

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print ("transformed scale:")
print (y.shape)
print (X.shape)

# Part 2: create ANN and fit data
def baseline_model():
    # Intialize the artificial neural network
    model = Sequential()

    # Input layer and hidden layer 
    model.add(Dense(activation="relu", input_dim=1191, units=700, kernel_initializer="glorot_normal"))
    # Dropout to avoid overfitting
    model.add(Dropout(0.3))
    
    # when >2 early stopping, when > E-5 flattening
    # model.add(ActivityRegularization(l1=0.005, l2=0.005))
    
    # add another smaller layer
    model.add(Dense(2000, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.3))
    
   
    # add another smaller layer // Test accuracy: 0.7252 (2000 epochs)
    # model.add(Dense(400, kernel_initializer='uniform', activation='tanh'))
    # model.add(Dropout(0.285))    
 
    # Output layer
    model.add(Dense(activation="relu", input_dim=700, units=400, kernel_initializer="uniform"))
     
    # Compile the ANN // loss="mean_squared_error" // loss="cosine_proximity"
    model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=["accuracy","mean_squared_error"])
    
    return model

# Keras callback save best models
# monitor for ['loss', 'acc', 'mean_squared_error', 'val_loss', 'val_acc', 'val_mean_squared_error']
checkpoint = ModelCheckpoint(filepath="best-model.hdf5",
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True)
                               
# Follow trends using tensorboard
# use source activate tensorflow
# start with tensorboard --logdir=logs
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

# enable eraly stopping
earlystopping=EarlyStopping(monitor='mean_squared_error', patience=200, verbose=1, mode='auto')

# Fit the ANN to the training set
model = baseline_model()

# calculate results, add callbacks such as tensorboard if needed
result = model.fit(X_train, y_train, batch_size=20, 
                   epochs=6000, validation_data=(X_test, y_test),
                   callbacks=[earlystopping,checkpoint])


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

# diagnostic plot model summary
model.summary()

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


# Part 4: export predictions
