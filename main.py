# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""

import MLATlib as lib
from MLATlib.helper import SP2CART

import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

if False:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')


# ### import and pre-process
use_pickle = True
use_file = 4  # -1 --> competition; 1 through 7 --> training

MR, NR, SR = lib.read.importData(use_pickle, use_file)
# print("Finished importing data\n")


# use separate SR file for validation
# or
# select random data points with GT from MR set
np.random.seed(2)
use_SR = False
K = 200000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation


# TRA, VAL = lib.read.segmentData(MR, use_SR, SR, K=K, p=p_vali)
TRA, VAL = lib.read.segmentDataByAC(MR, K, p_vali)


"""
### fakes for debugging
# fake nodes
node_sph = np.array([[50 ,11, 0],\
                     [50 ,9, 0],\
                     [51, 10, 0],\
                     [49, 10, 0]])

NR, idx_fake_n = lib.read.insertFakeStations(NR, node_sph)


# fake planes to training set
plane_sph = np.array([[50.1, 10.2, 2000]     , [56, 10.21, 0]])
plane_n   =           [tuple(idx_fake_n), tuple(idx_fake_n)]

TRA, idx_fake_planes = lib.read.insertFakePlanes(
    TRA, NR,
    plane_sph,
    plane_n,
    noise_amp=50
    )

VAL.loc[idx_fake_planes[0], ['lat', 'long', 'geoAlt']] = \
    TRA.loc[idx_fake_planes[0], ['lat', 'long', 'geoAlt']]

TRA.loc[idx_fake_planes[0], ['lat', 'long', 'geoAlt']] = np.nan
"""


"""# single plane stuff
# #######################

# select measurement to compute stuff for
# seek_id = 9999999 # fake plane
# seek_id = 111376 # some actually existing plane
# seek_id = 254475  # 3 stations, simple case
# seek_id = 1766269  # 4 stations, plane outside of interior
# seek_id = 880317 #  6 stations
# seek_id = 1857132  # 2 close stations mess it up
# seek_id = 1820733  # 2 close stations mess it up
# seek_id = 1028881  # 2 close stations mess it up
# seek_id = 1524975  # unknown convergence error
# seek_id = 29421  # hot mess..
seek_id = 503201  # best fit
# seek_id = 1823621  # 2 close stations mess it up
# seek_id = 1333057

# start MLAT calculations
x_sph, inDict = lib.ml.NLLS_MLAT(MR, NR, seek_id)

pp = lib.plot.HyperPlot(MR, SR, NR, seek_id, x_sph, inDict, SQfield=True)
"""


# itterazione
# ##############

# initialise solution dataframe
SOL = VAL.copy(deep=True)
SOL[["lat", "long", "geoAlt"]] = np.nan

t = time.time()
# pr = cProfile.Profile()
# pr.enable()
for idx in tqdm(SOL.index):
    try:
        xn_sph, inDict = lib.ml.NLLS_MLAT(TRA, NR, idx, solmode=1)
        SOL.loc[idx, ["lat", "long", "geoAlt"]] = xn_sph

        la, lo, al = zip(VAL.loc[idx])
        x_GT = SP2CART(la[0], lo[0], al[0])

        if len(inDict):
            fval_GT = lib.ml.FJsq(x_GT, inDict['A'], inDict['b'],
                                  inDict['dim'], inDict['V'],
                                  inDict['RD'], inDict['Rn'], mode=-1)
            TRA.at[idx, 'fval_GT'] = np.sum(fval_GT**2)

    except lib.ml.MLATError:
        pass
    # except (lib.ml.FeasibilityError, lib.ml.ConvergenceError):
    #    xn = np.array([-1, -1, -1])
    #    xn_sph = np.array([-1, -1, -1])
    #    fval = np.array([-1, -1, -1])
    #    pass

el = time.time() - t
print("\nTime taken: %f sec\n" % el)

# pr.disable()

RMSE, nv = lib.out.twoErrorCalc(SOL, VAL, RMSEnorm=2)

TRA.loc[VAL.index, "NormError"] = nv
SEL = TRA.loc[~np.isnan(TRA.NormError)]\
    .sort_values(by="NormError", ascending=True)

print(RMSE)
print(100*sum(nv > 0) / len(SOL))

lib.plot.ErrorCovariance(SEL)
lib.plot.ErrorHist(SEL)


SOL2 = SOL.copy()
acs = np.unique(TRA.loc[SOL.index, 'ac'])
for ac in acs:
    cur_id = TRA.loc[TRA['ac'] == ac].index
    cur_id = SOL2.index.intersection(cur_id)
    tempSOL = SOL2.loc[cur_id, 'long']
    cur_nonans = tempSOL.index[~np.isnan(tempSOL)]

    t = TRA.loc[cur_id, 't']
    t_nonan = t[cur_nonans].to_numpy()

    long = SOL.loc[cur_nonans, 'long'].to_numpy()
    lat = SOL.loc[cur_nonans, 'lat'].to_numpy()

    if len(lat):
        SOL2.loc[cur_id, 'long'] = np.interp(t, t_nonan, long,
                                             left=np.nan, right=np.nan)
        SOL2.loc[cur_id, 'lat'] = np.interp(t, t_nonan, lat,
                                            left=np.nan, right=np.nan)
    
        TRA.loc[cur_id, ['long', 'lat']] = SOL2.loc[cur_id, ['long', 'lat']]


lib.out.writeSolutions("../Comp1_.csv", SOL2)
# lib.out.writeSolutions("../Train7_.csv", SOL2)
RMSE, nv = lib.out.twoErrorCalc(SOL2, VAL, RMSEnorm=2)

TRA.loc[VAL.index, "NormError"] = nv
SEL = TRA.loc[~np.isnan(TRA.NormError)]\
    .sort_values(by="NormError", ascending=True)

print(RMSE)
print(100*sum(nv > 0) / len(SOL2))


lib.plot.ErrorCovariance(SEL)
lib.plot.ErrorHist(SEL)



# pp = lib.plot.PlanePlot()
# pp.addTrack(TRA, acs, z=VAL)




