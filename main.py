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
K     = 50

import readLib   as rlib
import outLib    as olib 
import MLAT      as ml


MR = rlib.readMeasurements("../training_1_round_1/training_1_category_1/training_1_category_1.csv", K)
NR = rlib.readNodes("../training_1_round_1/training_1_category_1/sensors.csv")
SR = rlib.readSolutions("../training_1_round_1_result/training_1_category_1_result/training_1_category_1_result.csv")



id_GT = MR[~np.isnan(MR.lat)].id # where we have ground truth from training set
Mrows_GT = len(id_GT)


# add fake beacons for debugging
idx_end = NR.n.iloc[-1]
NR = NR.append({"n" : int(idx_end+1), "lat" : 0,  "long" : 0,  "geoAlt" : 0,  "type" : "Test 1"}, ignore_index=True)
NR = NR.append({"n" : int(idx_end+2), "lat" : 0,  "long" : 45, "geoAlt" : 0,  "type" : "Test 2"}, ignore_index=True)
NR = NR.append({"n" : int(idx_end+3), "lat" : 0,  "long" : 90, "geoAlt" : 0,  "type" : "Test 3"}, ignore_index=True)
NR = NR.append({"n" : int(idx_end+4), "lat" : 90, "long" : 0,  "geoAlt" : 0,  "type" : "Test 4"}, ignore_index=True)

# add fake plane
node_cart = np.array([[X(0,0,0), Y(0,0,0), Z(0,0,0) ],\
                      [X(0,45,0),Y(0,45,0),Z(0,45,0)],\
                      [X(0,90,0),Y(0,90,0),Z(0,90,0)],\
                      [X(90,0,0),Y(90,0,0),Z(90,0,0)]])
cart_loc = np.sqrt(3)/3*6371e3*np.array([1,1,1])
ns       = np.vectorize(int)(la.norm(node_cart - cart_loc, axis=1)/C0*1e9)

MR = MR.append({"id": int(1e7-1), "t": 1e3-1, "ac": int(1e4-1), "lat": 0, "long": 0, \
                "baroAlt": 0, "geoAlt": -R0, "M": int(4), "m": "blah", \
                "n": idx_end+np.array([1,2,3,4]), \
                "ns": ns, "R": np.array([0, 0, 0, 0])}, ignore_index=True)

    
    
#ml.NLLS_MLAT(MR,NR,1439661)
ml.NLLS_MLAT(MR,NR,9999999)


#olib.writeSolutions("../training_1_round_1_result/test_out.csv", SR)