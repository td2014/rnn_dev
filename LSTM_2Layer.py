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

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
# End:  Set up environment for reproduction of results

#
from keras.layers import LSTM, Input, Masking, multiply
from keras.models import Model

#
# Create input sequences
#
numTimesteps=20
slopeArray1=np.linspace(0, 10, num=numTimesteps)
slopeArray1 = np.expand_dims(slopeArray1, axis=0)
slopeArray1 = np.expand_dims(slopeArray1, axis=2)

slopeArray2=np.linspace(0, 15, num=numTimesteps)
slopeArray2 = np.expand_dims(slopeArray2, axis=0)
slopeArray2 = np.expand_dims(slopeArray2, axis=2)
maskArray=np.zeros((1,numTimesteps,1))
maskArray[0,numTimesteps-1]=1

X_train = np.concatenate((slopeArray1, slopeArray2))
X_mask = np.concatenate((maskArray, maskArray))

# preparing y_train
y_train = []
y_train = np.array([2*slopeArray1[0,19]-slopeArray1[0,18],
                    2*slopeArray2[0,19]-slopeArray2[0,18]]) # make target one delta higher

#
# Create model
#

inputs = Input(name='Input1', batch_shape=(1,numTimesteps,1))
X_mask_input = Input(name='Input2', batch_shape=(1,numTimesteps,1))
x = LSTM(units=1, name='LSTM1', return_sequences=True)(inputs)
x = multiply([x, X_mask_input])
x = Masking(mask_value=0.0)(x)
pred = LSTM(units=1, name='LSTM2', return_sequences=False, stateful=True)(x)
model = Model(inputs=[inputs, X_mask_input], outputs=pred)
model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
print(model.summary())

#
# Train
# 
model.fit([X_train, X_mask], y_train, epochs=200, batch_size=1)

#
# output predictions
#
#print(model.get_weights()[0][0])
#model_predictions = model.predict(X_train)
#print('model_predictions = ', model_predictions)

#
# Other diagnostics
#

#
# End of script
#
