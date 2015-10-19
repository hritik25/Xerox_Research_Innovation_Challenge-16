# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 01:22:30 2015

@author: anelk-var
"""
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

DEBUG = True

# Read data TODO : add survival labels to this data
df= pd.read_csv('id_variables_combined_train.csv')
# test on small data for debug
if DEBUG:
    df = df[1:500]
    
df.fillna(0, inplace=True)

nbins = 500
timestamps = df['TIME']
timebins = np.histogram(timestamps, nbins)[0]

# split data by survival labels added above and do
# following for each separately
for label in [True, False]:
    dfsub = df[df['SURVIVE']==label]
    values = []
    for i in np.unique(timebins):
        # sum values in one bin
        indices = np.where(timebins==i)
        values.append(dfsub.iloc[indices].sum()/len(indices))
    # plot values
        
        