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


#%% import

use_pickle = False
use_file = -1  # -1 --> competition; 1 through 7 --> training

MR, NR, SR = lib.read.importData(use_pickle, use_file)


#%% segment data

# use separate SR file for validation
# or
# select random data points with GT from MR set
np.random.seed(2)
use_SR = True
K = 20000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation


TRA, VAL = lib.read.segmentData(MR, use_SR, SR, K=K, p=p_vali)
# TRA, VAL = lib.read.segmentDataByAC(MR, K, p_vali)

# print("Finished importing data\n")


#%% fakes for debugging
"""
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


#%% single plane stuff
"""

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


#%% initialize

# clock corrector
alpha = 0.2
NR_c = lib.sync.NR_corrector(TRA, NR, alpha)

# initialise solution dataframe
SOL = VAL.copy(deep=True)
SOL[["lat", "long", "geoAlt"]] = np.nan

# initualize np arrays
xn_sph_np = np.zeros([len(SOL), 3])
xn_sph_np[:, :] = np.nan

# ground truth as np
lats, longs, alts = VAL.to_numpy().T
x_GT = SP2CART(lats, longs, alts).T
fval_GT = np.zeros(len(SOL))
fval_GT[:] = np.nan


#%% itterazione

t = time.time()

npi = 0
for idx, row in tqdm(TRA.iterrows(), total=len(TRA)):
    if (SOL.index == idx).any():
        try:
            assert(idx > 8*60/3600*len(TRA))
            
            xn_sph_np[npi], inDict = lib.ml.NLLS_MLAT(TRA, NR, idx, NR_c, 
                                                      solmode='2d')

            if len(inDict):
                fval_GT[npi] = lib.ml.FJsq(x_GT[npi],
                                           inDict['A'], inDict['b'],
                                           inDict['dim'], inDict['V'],
                                           inDict['RD'], inDict['Rn'],
                                           mode=0
                                           )

        except (lib.ml.MLATError, AssertionError):
            pass
        finally:
            npi += 1
    else:
        # do a relative sync
        NR_c.RelativeSync(row, idx)

        cnt = 1
        if (row['t'] / 300) > cnt:
            NR_c.AbsoluteSync()
            cnt += 1

SOL[["lat", "long", "geoAlt"]] = xn_sph_np
TRA.loc[SOL.index, ['lat', 'long', 'geoAlt']] = xn_sph_np

el = time.time() - t
print("\nTime taken: %f sec\n" % el)


#%% Prune to trustworthy data

fval_thres = 1e8
TRA_temp, SOL_temp = lib.ml.PruneResults(TRA, SOL, fval_thres)


#%% Print out intermediate accuracy

RMSE, cov, nv = lib.out.twoErrorCalc(SOL_temp, VAL, RMSEnorm=2)

print(RMSE)
print(cov*100)


#%% do the filtering and interpolation

TRA_temp['MLAT'] = False
TRA_temp.loc[TRA_temp['MLAT_status'] == 0, 'MLAT'] = True

TRA2 = TRA_temp.copy()
TRA2['score'] = np.nan
SOL2 = SOL_temp.copy()
acs = np.unique(TRA2.loc[SOL2.index, 'ac'])
for ac in tqdm(acs):
    aco = lib.filt.aircraft(TRA_temp, SOL_temp, ac)
    aco.Interp(usepnts='adaptive')
    SOL2.loc[aco.ids] = aco.SOLac
    TRA2.loc[aco.ids] = aco.TRAac


#%% keep only best 50% by score

covEst = len(TRA2[~np.isnan(TRA2['score'])]) / len(SOL2)
keepPercentile = 0.51 / covEst
loseIndex = TRA2.index[TRA2['score'] > TRA2['score'].quantile(keepPercentile)]

TRA3 = TRA2.copy()
SOL3 = SOL2.copy()

TRA3.loc[loseIndex, ['lat', 'long', 'geoAlt']] = np.nan
SOL3.loc[loseIndex, ['lat', 'long', 'geoAlt']] = np.nan


#%% Print final accuracy

RMSE, cov, nv = lib.out.twoErrorCalc(SOL3, VAL, RMSEnorm=2)

print(RMSE)
print(cov*100)


#%% write

lib.out.writeSolutions("../Comp1_.csv", SOL3)
# lib.out.writeSolutions("../Train7_.csv", SOL)


#%% sort the final data frames and append with some GT data

TRA3.loc[SOL3.index, 'fval_GT'] = fval_GT
TRA3.loc[VAL.index, "NormError"] = nv
SEL = TRA3.loc[~np.isnan(TRA3.NormError)]\
    .sort_values(by="NormError", ascending=True)


# !!! NO MORE ACCURACY-IMPROVING CODE AFTER HERE!!! It would be cheating


#%% plotting

# Error plots
# lib.plot.ErrorCovariance(SEL)
# lib.plot.ErrorHist(SEL)

# Track plots
pp = lib.plot.PlanePlot()
pp.addTrack(TRA3, acs, z=VAL, color='orange')
# pp.addTrack(TRA_temp, acs, z=VAL, color='orange')
# pp.addTrack(TRA2, [2213], z=VAL, color='orange')
# pp.addTrack(TRA, [2213], z=VAL)


#%% EOF
