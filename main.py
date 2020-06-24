# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

from   constants import *

import              numpy  as np
from   numpy import linalg as la

import pandas as pd

# set random number and K
import random as rd
rd.seed(1)
K     = 50 # how many data points to read out of the millions

import readLib   as rlib
import outLib    as olib 
import MLAT      as ml




### import and pre-process

# read csv files
MR = rlib.readMeasurements("../training_1_round_1/training_1_category_1/training_1_category_1.csv", K)
NR = rlib.readNodes("../training_1_round_1/training_1_category_1/sensors.csv")
SR = rlib.readSolutions("../training_1_round_1_result/training_1_category_1_result/training_1_category_1_result.csv")

# find datapoints with ground truth inside training set
id_GT = MR[~np.isnan(MR.lat)].id # where we have ground truth from training set
Mrows_GT = len(id_GT)





### define and add fake stations for debugging

# define fake stations in spherical
node_sp = np.array([[50 ,11, 0],\
                    [50 ,9, 0],\
                    [51, 10, 0],\
                    [49, 10, 0]])
    

# add fake stations
idx_end = NR.n.iloc[-1]
NR = NR.append({"n" : int(idx_end+1), "lat" : node_sp[0,0],  "long" : node_sp[0,1], "geoAlt" : node_sp[0,2],  "type" : "Test 1"}, ignore_index=True)
NR = NR.append({"n" : int(idx_end+2), "lat" : node_sp[1,0],  "long" : node_sp[1,1], "geoAlt" : node_sp[1,2],  "type" : "Test 2"}, ignore_index=True)
NR = NR.append({"n" : int(idx_end+3), "lat" : node_sp[2,0],  "long" : node_sp[2,1], "geoAlt" : node_sp[2,2],  "type" : "Test 3"}, ignore_index=True)
NR = NR.append({"n" : int(idx_end+4), "lat" : node_sp[3,0],  "long" : node_sp[3,1], "geoAlt" : node_sp[3,2],  "type" : "Test 4"}, ignore_index=True)

# convert nodes to cartesian for calculating TOAs
node_cart = SP2CART(node_sp[:,0], node_sp[:,1], node_sp[:,2]).T





### define and add fake plane for debugging

# define fake plane
sp_loc   = np.array([56, 10.2, 0])


# convert to cartesian
cart_loc = SP2CART(sp_loc[0], sp_loc[1], sp_loc[2])

# find nanoseconds TOA (in this case equal to TOT, since no offset)
ns       = np.vectorize(int)(la.norm(node_cart - cart_loc, axis=1)/C0*1e9)
#print("actual TOA:", ns)

# add fake plane
MR = MR.append({"id": int(1e7-1), "t": 1e3-1, "ac": int(1e4-1), "lat": sp_loc[0], "long": sp_loc[1], \
                "baroAlt": sp_loc[2], "geoAlt": sp_loc[2], "M": int(4), "m": "blah", \
                "n": idx_end+np.array([1,2,3,4]), \
                "ns": ns, "R": np.array([0, 0, 0, 0])}, ignore_index=True)


    
### actually do stuff

# select measurement to compute stuff for
#seek_id = 9999999 # fake plane
seek_id = MR.iloc[15].id # some actually existing plane

# start MLAT calculations
c, found_loc, fval = ml.NLLS_MLAT(MR,NR,seek_id)

# print result
print(np.array([found_loc,[MR[MR.id == seek_id].lat.iloc[0],\
                MR[MR.id == seek_id].long.iloc[0],\
                MR[MR.id == seek_id].geoAlt.iloc[0]] ]))

# plotting
pp = olib.PlanePlot()
pp.addPoint(MR, [seek_id])
pp.addPointByCoords(np.array([found_loc[0:2]]))
pp.addNodeById(NR, MR, [seek_id])
#olib.writeSolutions("../training_1_round_1_result/test_out.csv", SR)