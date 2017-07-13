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
from keras.callbacks import Callback

#
# Create input sequences
#
X_train = []
up_start = 0
up_end = 7
down_start=2000
down_end=1000
up_array = np.linspace(up_start,up_end,int(up_end-up_start))
upsweep=np.array(up_array)-(up_end+up_start)/2.0
upsweep=upsweep/(1.0*np.max((up_start,up_end)))
upsweep = np.expand_dims(upsweep,axis=1)

down_array = np.linspace(down_start,down_end,int(down_start-down_end))
downsweep=np.array(down_array)-(down_start+down_end)/2.0
downsweep=downsweep/(1.0*np.max((down_start,down_end)))
downsweep = np.expand_dims(downsweep,axis=1)

X_train.append(upsweep)
###X_train.append(downsweep)
###X_train.append(downsweep)
#X_train.append(upsweep)
#X_train.append(downsweep)
#X_train.append(downsweep)
#X_train.append(upsweep)
#X_train.append(downsweep)
#X_train.append(upsweep)

X_train = np.array(X_train)

# preparing y_train
y_train = []
y_train.append([10,11,12,13,14,15,16])

y_train = np.array(y_train)
y_train = np.expand_dims(y_train,axis=2)

#
# Create model
#

the_inputs = Input(shape=(7,1), name='Input1')
x = LSTM(units=1, name='LSTM1', return_sequences=True)(the_inputs)
x = Dense(2)(x)
predictions = Dense(1, name='Dense1')(x)
model = Model(inputs=the_inputs, outputs=predictions)
model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#
# Train
# 

for iEpoch in range(1):
    
    print('iEpoch = ', iEpoch)
    print('X_train = ', X_train)
    print('y_train = ', y_train)
    print('Model.weights before training = ', model.get_weights()[0][0])
    model.fit(X_train, y_train, epochs=1, batch_size=1)
    print('Model.weights after training = ', model.get_weights()[0][0])
    model_output = model.predict(X_train)
    print('model_output= ', model_output)
    y_train[0] = y_train[0]+1
    
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
