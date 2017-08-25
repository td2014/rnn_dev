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
from keras.layers import LSTM, Dense, Input
from keras.models import Model

#
# Create input sequences
#
X_train = []
up_start = 0
up_end = 1000
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
X_train.append(downsweep)
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
y_train.append(2*upsweep[999]-upsweep[998])
y_train.append(2*downsweep[999]-downsweep[998])
###y_train.append([3])
#y_train.append([0,1])
#y_train.append([1,0])
#y_train.append([0,1])
#y_train.append([0,1])
#y_train.append([1,0])
#y_train.append([0,1])
#y_train.append([1,0])

y_train = np.array(y_train)

#
# Create model
#

inputs = Input(shape=(1000,1), name='Input1')
x = LSTM(units=1, name='LSTM1')(inputs)
x = Dense(2)(x)
predictions = Dense(1, name='Dense1')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#
# Train
# 
print('Training model...')
model.fit(X_train, y_train, epochs=500, batch_size=2)

#
# output predictions
#
print(model.get_weights()[0][0])
model_predictions = model.predict(X_train)
print('model_predictions = ', model_predictions)

#
# Other diagnostics
#
#print()
#model.evaluate(X_train, y_train)


#
# End of script
#
