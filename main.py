# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""

from constants import R0, X, Y, Z, SP2CART, CART2SP


import numpy as np
# from numpy import linalg as la

import pandas as pd

import readLib as rlib
import outLib as olib
import MLAT_hyp as ml

import os
import time

from tqdm import tqdm

if False:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')


# ### import and pre-process
use_pickle = True
use_file = 1  # -1 --> competition; 1 through 7 --> training

MR, NR, SR = rlib.importData(use_pickle, use_file)
# print("Finished importing data\n")


# use separate SR file for validation
# or
# select random data points with GT from MR set
np.random.seed(1)
use_SR = False
K = 10000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation

TRA, VAL = rlib.segmentData(MR, use_SR, SR, K=K, p=p_vali)



"""
### fakes for debugging
# fake nodes
node_sph = np.array([[50 ,11, 0],\
                     [50 ,9, 0],\
                     [51, 10, 0],\
                     [49, 10, 0]])

NR, idx_fake_n = rlib.insertFakeStations(NR, node_sph)


# fake planes to training set
plane_sph = np.array([[56, 10.2, 0]     , [56, 10.21, 0]])
plane_n   =           [tuple(idx_fake_n), tuple(idx_fake_n)]

TRA, idx_fake_planes = rlib.insertFakePlanes(TRA, NR,
                                             np.array([[1,2,3], [2,3,4]]),
                                             [(523,), (522,523)],
                                             noise_amp = 10)
"""


### single plane stuff
# select measurement to compute stuff for
#seek_id = 9999999 # fake plane
seek_id = 1141465 # some actually existing plane

# start MLAT calculations
c, found_loc, fval = ml.NLLS_MLAT(MR, NR, seek_id)

# print result
print(np.array([found_loc,
                [MR.at[seek_id, 'lat'],
                 MR.at[seek_id, 'long'],
                 MR.at[seek_id, 'geoAlt']
                 ]
                ]))

# plotting
pp = olib.PlanePlot()
pp.addPoint(MR, [seek_id])
pp.addPointByCoords(np.array([found_loc[0:2]]))
pp.addNodeById(NR, MR, [seek_id])
#pp.addTrack(MR, [MR.at[seek_id, 'ac']])
olib.writeSolutions("./test_out.csv", SR)

"""

# initialise solution dataframe
SOL = VAL.copy(deep=True)
SOL[["lat", "long", "geoAlt"]] = np.nan

t = time.time()
interv = 250
counter = 1
maxcount = len(SOL)
# pr = cProfile.Profile()
# pr.enable()
for idx in tqdm(SOL.index):
    try:
        xn, xn_sph, fval = ml.NLLS_MLAT(TRA, NR, idx, solmode=1)
        SOL.loc[idx, ["lat", "long", "geoAlt"]] = xn_sph

    except (ml.FeasibilityError, ml.ConvergenceError):
        xn = np.array([-1, -1, -1])
        xn_sph = np.array([-1, -1, -1])
        fval = np.array([-1, -1, -1])
        pass

    counter = counter + 1

el = time.time() - t
print("\nTime taken: %f sec\n" % el)

# pr.disable()

olib.writeSolutions("../Training7_9e68d8.csv", SOL)
RMSE, nv = olib.twoErrorCalc(SOL, VAL, RMSEnorm=2)

TRA.loc[VAL.index, "NormError"] = nv
SEL = TRA.loc[~np.isnan(TRA.NormError)]\
    .sort_values(by="NormError", ascending=True)

print(RMSE)
print(100*sum(nv > 0) / maxcount)
"""