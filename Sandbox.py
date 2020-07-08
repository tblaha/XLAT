# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:21:17 2020

@author: Till
"""



import numpy as np
from mayavi import mlab
#from main import MR, NR, SR
from constants import *
import outLib as olib
import MLAT as ml
import sys
import matplotlib.pyplot as plt

ac = 971

MR_sub = MR[MR.ac == ac].reset_index().drop(columns=["index"])
num = MR_sub.shape[0]
stations = np.unique(np.concatenate(MR_sub.n.to_numpy()))
num_st = stations.shape[0]

T_stations = np.empty([num_st, num])
T_stations[:] = np.NaN

for index, row in MR_sub.iterrows():
    
    for (n, ns) in zip(row.n, row.ns):
        T_stations[np.where(stations == n)[0][0], index] = ns
    
T_stations[T_stations == 0] = np.nan

plt.close("all")
plt.figure()
for i in np.arange(num_st):
    plt.plot(MR_sub.t,\
             T_stations[i,:,np.newaxis] - MR_sub.t[:, np.newaxis]*1e9, \
             '.', label="No. %d"%stations[i])

plt.legend(loc="upper right")
plt.grid()