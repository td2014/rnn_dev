#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 05:25:10 2017

@author: anthonydaniell
"""
def firstn(n):
    num = 0
    while num < n:
       yield num
       num += 1

#
# End of script
#