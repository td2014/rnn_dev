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
from keras.layers import LSTM, Dense, Input
from keras.models import Model

#
# Create input sequences
#
X_train = []

upsweep=np.array((1,2,3,4,5,6,7))
upsweep = np.expand_dims(upsweep,axis=1)
downsweep=np.array((10, 9, 8, 7, 6, 5, 4))
downsweep = np.expand_dims(downsweep,axis=1)

X_train.append(upsweep)
X_train.append(downsweep)
X_train.append(downsweep)
X_train.append(upsweep)
X_train.append(downsweep)
X_train.append(downsweep)
X_train.append(upsweep)
X_train.append(downsweep)
X_train.append(upsweep)
X_train = np.array(X_train)


# preparing y_train
y_train = []
y_train.append([1,0])
y_train.append([0,1])
y_train.append([0,1])
y_train.append([1,0])
y_train.append([0,1])
y_train.append([0,1])
y_train.append([1,0])
y_train.append([0,1])
y_train.append([1,0])

y_train = np.array(y_train)

#
# Create model
#

inputs = Input(shape=(7,1), name='Input1')
x = LSTM(units=1, name='LSTM1')(inputs)
predictions = Dense(2, activation='sigmoid', name='Dense1')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#
# Train
# 
print('Training model...')
model.fit(X_train, y_train, epochs=10)

#
# output predictions
#
##print(model.get_weights()[0][0])
##predictions = model.predict(X_train)

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
