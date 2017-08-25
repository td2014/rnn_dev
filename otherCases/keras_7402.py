#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:33:02 2017

@author: anthonydaniell
"""

from keras.layers import LSTM, Input, Dense
from keras.models import Model
import numpy as np
##import pydot
##import graphviz
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot


y = np.float32(np.reshape(np.random.normal(0, 1, 100), (100, )))
x = np.float32(np.reshape(np.array([0] * 100), (100, 1, 1)))

input_layer = Input(shape = (1, 1))
lstm_layer = LSTM(1)(input_layer)
dense_layer = Dense(1)(lstm_layer)
model = Model(input = input_layer, output = dense_layer)
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
model.fit(x, y, epochs = 1, batch_size = 1, verbose = 2)

plot_model(model, to_file='ald_model_v2.png')
###SVG(model_to_dot(model).create(prog='dot', format='svg'))


# -*- encoding: utf-8 -*-
"""
pydot graph example
@author: Francisco Portillo
@url: https://gist.github.com/fportillo/a499f9f625c6169524b8
"""
##import pydot
##import graphviz

# Create the graph
##graph = pydot.Dot(graph_type='digraph')

# Create nodes
##open = pydot.Node("Open")
##in_progress = pydot.Node("In progress")
##cancelled = pydot.Node("Cancelled")
##finished = pydot.Node("Finished")

##graph.add_node(open)
##graph.add_node(in_progress)
##graph.add_node(cancelled)
##graph.add_node(finished)

##graph.add_edge(pydot.Edge(open, in_progress))
##graph.add_edge(pydot.Edge(open, cancelled))
##graph.add_edge(pydot.Edge(in_progress, finished))
##graph.add_edge(pydot.Edge(in_progress, cancelled))

##graph.write_png("ald_pydot_example.png")
