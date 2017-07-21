#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:12:49 2017

@author: anthonydaniell
"""
# Start: Set up environment for reproduction of results
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
#single thread
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# End:  Set up environment for reproduction of results

#
from keras.layers import LSTM, Dense, Input, SimpleRNN
from keras.models import Model

#
# Create input sequences
#
X_train = []
up_start = 1
up_end = 8
up_array = np.linspace(up_start,up_end,int(up_end-up_start+1))
up_array2 = np.linspace(up_start,up_end,int(up_end-up_start+1))
up_array2 = up_array2*3
up_final = np.stack((up_array,up_array2))
up_final = up_final.transpose()
upsweep=np.array(up_final)

X_train.append(upsweep)
X_train = np.array(X_train)

# preparing y_train
y_train = []
y_train.append([9,27]) # targets
y_train = np.array(y_train)

#
# Create model
#

the_inputs = Input(shape=(8,2), name='Input1')
x = LSTM(units=2, name='LSTM1', return_sequences=False, 
         return_state=False, activation='linear')(the_inputs)
model = Model(inputs=the_inputs, outputs=x)
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#
# Train
#
print('X_train = ', X_train)
print('y_train = ', y_train) 

for iEpoch in range(200):
    
    print('iEpoch = ', iEpoch)
    print('Model.weights before training = ', model.get_weights()[0][0])
    model.fit(X_train, y_train, epochs=1, batch_size=1)
    print('Model.weights after training = ', model.get_weights()[0][0])
    model_output = model.predict(X_train)
    print
    print('model_output= ', model_output)
    
#update target here


#
# output predictions
#
##print(model.get_weights()[0][0])
##model_predictions = model.predict(X_train)
##print('model_predictions = ', model_predictions)

#
# Other diagnostics
#
#print()
#model.evaluate(X_train, y_train)

#
# Ref values for one epoch:   [0.24794224, 0.66666669]
#

#
# End of script
#
