# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 19:55:19 2018

@author: nrgurram
"""

MIS = {1:5, 2:4, 3:3, 4:2, 5:1};
items = sorted([1,2,3,4,5], key=lambda x : (MIS[x], x));
print(str(items))