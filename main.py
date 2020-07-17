# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')




import cProfile, pstats, io
from pstats import SortKey

from   constants import *


import              numpy  as np
from   numpy import linalg as la

import pandas as pd

import readLib   as rlib
import outLib    as olib 
import MLAT      as ml

import os
import time

import sklearn.model_selection as sklms

import tqdm




###import and pre-process
use_pickle = False
use_files  =  7 # -1 --> competition; 1 through 7 --> training

if use_files == -1:
    path    = "../Data/round1_competition"
    fnameMR = "round1_competition.csv"
    fnameSR = "round1_sample_empty.csv"
elif use_files > 0:
    path    = "../Data/training_"+str(use_files)+"_category_1"
    fnameMR = "training_"+str(use_files)+"_category_1.csv"
    fnameSR = "training_"+str(use_files)+"_category_1_result.csv"



    
# read csv files
if os.path.isfile("./MR.pkl") and use_pickle:
    MR = pd.read_pickle("./MR.pkl")
else:
    MR = rlib.readMeasurements(path+"/"+fnameMR)
    MR.to_pickle("./MR.pkl")



if os.path.isfile("./NR.pkl") and use_pickle:
    NR = pd.read_pickle("./NR.pkl")
else:
    NR = rlib.readNodes(path+"/sensors.csv")
    NR.to_pickle("./NR.pkl")
    
    
    
if os.path.isfile("./SR.pkl") and use_pickle:
    SR = pd.read_pickle("./SR.pkl")
else:
    SR = rlib.readSolutions(path+"_result/"+fnameSR)
    SR.to_pickle("./SR.pkl")



print("Finished importing data\n")





# use separate SR file for validation 
# or
# select random data points with GT from MR set
np.random.seed(1)
use_SR = True
K      = 10000 # how many data points to read out of the millions and use for validation
p_vali = 0.05 # share of K used for validation

TRA, VAL = rlib.segmentData(MR, use_SR, SR, K = K, p = p_vali)

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

TRA, idx_fake_planes = rlib.insertFakePlanes(TRA, NR, np.array([[1,2,3], [2,3,4]]), [(523,), (522,523)], noise_amp = 10)
"""

"""
### single plane stuff
# select measurement to compute stuff for
#seek_id = 9999999 # fake plane
seek_id = 1169065 # some actually existing plane

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
pp.addTrack(MR, [MR.at[seek_id, 'ac']])
olib.writeSolutions("./test_out.csv", SR)
"""


# initialise solution dataframe
SOL = VAL.copy(deep=True)
SOL[["lat", "long", "geoAlt"]] = np.nan




t = time.time()
interv = 250
counter = 1
maxcount = len(SOL)
#pr = cProfile.Profile()
#pr.enable()
for idx in tqdm(SOL.index):
    try:
        xn, xn_sph, fval = ml.NLLS_MLAT(TRA, NR, idx, solmode = 1)
        SOL.loc[idx, ["lat", "long", "geoAlt"]] = xn_sph
        
    except (ml.FeasibilityError, ml.ConvergenceError):
        #xn     = np.array([-1, -1, -1])
        #xn_sph = np.array([-1, -1, -1])
        #fval   = np.array([-1, -1, -1])
        pass
    
    if not counter%interv:
        print("current count:", counter, " of ", maxcount, ": %.2f %%" % (counter/maxcount*100) )
        cur_time = time.time() - t
        print("Estimated time to go: %d" % (cur_time/(counter/maxcount) - cur_time) )
    
    counter = counter + 1
    
el = time.time() - t
print("\nTime taken: %f sec\n"%el)


#pr.disable()

olib.writeSolutions("../Training7_.csv", SOL)
RMSE, nv = olib.twoErrorCalc(SOL, VAL, RMSEnorm = 2)

TRA.loc[VAL.index, "NormError"] = nv[0]
SEL = TRA.loc[~np.isnan(TRA.NormError) & TRA.NormError >= 3000].sort_values(by="NormError", ascending=True)

print(RMSE)
print(100*sum( (nv > 0)[0] ) / maxcount)