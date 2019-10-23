# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:02:27 2018

@author: nrgurram
"""

import pandas as pd
import numpy as np

'''
Bernouli distribution
'''
testdata = pd.read_csv("testdata.csv", header=None)
bernouli_data = testdata[9]
bernouli_param_estimate = np.mean(bernouli_data)
print(bernouli_param_estimate)

'''

'''