#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:29:56 2017

Recursive function experiment

@author: anthonydaniell
"""
import matplotlib.pyplot as plt

y = []
y.append(-2)

for iLoop in range(10):
    
    y.append(y[iLoop]+2)
    
 
#
# Create model:
#    y(t) = a * y(t-1)
#

y = []
y.append(-1) #-2
y.append(-1)  #4
a = -0.8  # -0.8  causes oscillation, x = 0.276393, x = 0.723607
b = 0.2  # 0.2 causes oscillation

for iLoop in range(100):
    
    y.append(a*y[iLoop+1]+b*y[iLoop])    
    
plt.plot(y)
#
# End of script
#