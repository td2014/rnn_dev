#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:29:56 2017

Recursive function experiment

@author: anthonydaniell
"""

y = []
y.append(-2)

for iLoop in range(10):
    
    y.append(y[iLoop]+2)

#
# End of script
#