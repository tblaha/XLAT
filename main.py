# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""

from constants import R0, X, Y, Z, SP2CART, CART2SP, sind, cosd


import numpy as np
# from numpy import linalg as la

import pandas as pd

import readLib as rlib
import outLib as olib
import MLAT_hyp as ml

import os
import time

from tqdm import tqdm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

if False:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')


# ### import and pre-process
use_pickle = True
use_file = 4  # -1 --> competition; 1 through 7 --> training

MR, NR, SR = rlib.importData(use_pickle, use_file)
# print("Finished importing data\n")


# use separate SR file for validation
# or
# select random data points with GT from MR set
np.random.seed(1)
use_SR = True
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
plane_sph = np.array([[50.1, 10.2, 2000]     , [56, 10.21, 0]])
plane_n   =           [tuple(idx_fake_n), tuple(idx_fake_n)]

TRA, idx_fake_planes = rlib.insertFakePlanes(
    TRA, NR,
    plane_sph,
    plane_n,
    noise_amp=50
    )

VAL.loc[idx_fake_planes[0], ['lat', 'long', 'geoAlt']] = \
    TRA.loc[idx_fake_planes[0], ['lat', 'long', 'geoAlt']]

TRA.loc[idx_fake_planes[0], ['lat', 'long', 'geoAlt']] = np.nan
"""


# single plane stuff
# select measurement to compute stuff for
#seek_id = 9999999 # fake plane
# seek_id = 111376 # some actually existing plane
# seek_id = 254475  # 3 stations, simple case
#seek_id = 1766269  # 4 stations, plane outside of interior
seek_id = 880317 #  6 stations

# start MLAT calculations
#c, found_loc, fval = ml.NLLS_MLAT(MR, NR, seek_id)
t = time.time()
x_sph, inDict = ml.NLLS_MLAT(MR, NR, seek_id)
print(time.time() - t)

# print result

print(np.array([x_sph,
                [MR.at[seek_id, 'lat'],
                 MR.at[seek_id, 'long'],
                 MR.at[seek_id, 'geoAlt']
                 ]
                ]))

# plotting
pp = olib.PlanePlot()
pp.addPoint(SR, [seek_id])
pp.addPointByCoords(np.array([x_sph[0:2]]))
pp.addNodeById(NR, MR, [seek_id])
#pp.addTrack(MR, [MR.at[seek_id, 'ac']])
#olib.writeSolutions("./test_out.csv", SR)

n_vals = 100
longl, longu, latl, latu = pp.ax.get_extent()
long, lat = np.meshgrid(np.linspace(longl, 
                                    longu, n_vals),
                        np.linspace(latl,
                                    latu, n_vals)
                        )
r = R0 + TRA.at[seek_id, 'baroAlt']
x = r * cosd(long) * cosd(lat)
y = r * sind(long) * cosd(lat)
z = r *              sind(lat)

F   = np.zeros([inDict['dim'], n_vals, n_vals])
Fsq = np.zeros([n_vals, n_vals])
for i in range(n_vals):
    for j in range(n_vals):
        xvec = np.array([x[i, j], y[i, j], z[i, j]])
        F[:, i, j] = inDict['fun'](xvec, -1)
        Fsq[i, j] = np.sum(F[[0, 1, 2, 3, 4, 5, 6, 7, 9], i, j]**2)
        Fsq[i, j] = inDict['fun'](xvec, 0)

# plot global optimum
# cs = pp.ax.contour(long, lat, Fsq, np.logspace(1, 7, 30),
#                     transform=ccrs.PlateCarree(),
#                     )
#pp.ax.clabel(cs, fontsize=10)

# plot indivudial measurements
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
i = 0
#for i in range(inDict['dim']):
for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
    cs = pp.ax.contour(long, lat, F[i], [0], 
                        transform=ccrs.PlateCarree(), 
                        colors=colors[i%len(colors)],
                        )
    pp.ax.clabel(cs, fontsize=10)

# plot history
xhist = np.array(inDict["xlist"])
xhist_sph = CART2SP(xhist[:, 0], xhist[:, 1], xhist[:, 2])
pp.ax.plot(xhist_sph[1, :], xhist_sph[0, :],
           transform=ccrs.PlateCarree(),
           color='k',
           marker='.',
           )









"""
# initialise solution dataframe
SOL = VAL.copy(deep=True)
SOL[["lat", "long", "geoAlt"]] = np.nan

t = time.time()
# pr = cProfile.Profile()
# pr.enable()
for idx in tqdm(SOL.index):
    try:
        xn_sph, inDict = ml.NLLS_MLAT(TRA, NR, idx, solmode=1)
        SOL.loc[idx, ["lat", "long", "geoAlt"]] = xn_sph
        
        la, lo, al = zip(VAL.loc[idx])
        x_GT = SP2CART(la[0], lo[0], al[0])
        
        if len(inDict):
            fval_GT = ml.FJ(x_GT, inDict['A'], inDict['b'], inDict['dim'],
                            inDict['V'], inDict['RD'], mode=0)
            TRA.at[idx, 'fval_GT'] = np.sum(fval_GT**2)
        
    except ml.MLATError:
        pass
    #except (ml.FeasibilityError, ml.ConvergenceError):
    #    xn = np.array([-1, -1, -1])
    #    xn_sph = np.array([-1, -1, -1])
    #    fval = np.array([-1, -1, -1])
    #    pass

el = time.time() - t
print("\nTime taken: %f sec\n" % el)

# pr.disable()


#olib.writeSolutions("../Comp1_9e68d8.csv", SOL)
RMSE, nv = olib.twoErrorCalc(SOL, VAL, RMSEnorm=2)

TRA.loc[VAL.index, "NormError"] = nv
SEL = TRA.loc[~np.isnan(TRA.NormError)]\
    .sort_values(by="NormError", ascending=True)

print(RMSE)
print(100*sum(nv > 0) / len(SOL))
"""
