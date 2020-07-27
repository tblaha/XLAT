import MLATlib as lib
from MLATlib.helper import SP2CART, C0

import os
import time
from tqdm import tqdm

import numpy as np
import numpy.linalg as la
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib import pyplot as plt


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
K = 100000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation

# TRA, VAL = lib.read.segmentDataByAC(MR, K=K, p=p_vali)
TRA, VAL = lib.read.segmentData(MR, use_SR, SR=SR, K=K, p=p_vali)

TRA_GT = TRA.loc[~np.isnan(TRA['lat'])]

mps = list()
for n in range(max(TRA_GT['M']) + 1):
    mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
    mps.append(mp)

CART = np.zeros([3, len(NR)])
for index, row in NR.iterrows():
    lat, long, geoh \
        = NR.loc[index, ['lat', 'long', 'geoAlt']].to_numpy().T
    CART[:, index-1] = SP2CART(lat, long, geoh)

NR['x'] = CART[0, :]
NR['y'] = CART[1, :]
NR['z'] = CART[2, :]

NRnp = NR.loc[:, ['x', 'y', 'z']].to_numpy()
NR_corr = [list([[], [], [], 0]) for _ in range(len(NR))]
alpha = 0.4

for index, row in tqdm(TRA_GT.iterrows()):
    # nice breakpoints:
    # (Nids[rem] == 473) & (d < 0)
    # index == 201151
    # (Nids == 473).any()
    # index == 104513
    M = row.iat[6]
    if M < 3:
        continue

    mp = mps[M]
    Nids = np.array(row.iat[8])
    N = NRnp[Nids-1]
    t = row.iat[0]

    # x_sph_GT = row[['lat', 'long', 'geoAlt']].to_numpy()
    x_sph_GT = row.iloc[[2,3,4]].to_numpy()

    x_GT = SP2CART(x_sph_GT[0], x_sph_GT[1], x_sph_GT[2])

    Rs_GT = la.norm(x_GT - N, axis=1)
    RD_GT = (- Rs_GT[mp[:, 1]] + Rs_GT[mp[:, 0]])

    Rs_corr = np.array([NR_corr[i-1][3] for i in Nids])
    Rs = np.array(row.iat[9]) * 1e-9 * C0 + Rs_corr
    RD = (- Rs[mp[:, 1]] + Rs[mp[:, 0]])

    diff = RD_GT - RD
    # for i, d in enumerate(diff):
    #     if abs(d) < 1e7:
    #         NR_corr[Nids[mp[i, 0]]-1][0].append(t)
    #         NR_corr[Nids[mp[i, 0]]-1][1].append(d/2)
    #         NR_corr[Nids[mp[i, 1]]-1][0].append(t)
    #         NR_corr[Nids[mp[i, 1]]-1][1].append(-d/2)
    
    score = np.zeros([M])
    med = np.zeros([M])
    while sum((~np.isnan(diff)).astype(int)) > 1:
        for i in range(M):
            x = np.abs(diff[(mp == i).any(axis=1)])
            v = np.nanvar(x)**0.5
            med[i] = np.nanmedian(x)
    
            score[i] = v
        
        # if np.nanmin(score) > 500:
        #     # print("nope")
        #     break
        
        rem = np.nanargmin(score)
        
        diffidx = np.where( (mp == rem).any(axis = 1) 
                           & (~np.isnan(diff)) )[0][-1]
        
        cond = mp[diffidx, 0] == rem
        flip = 1 if cond else -1
        
        d = np.sign(diff[diffidx]) * flip * med[rem]
        
        if abs(d) < 1e6:
            cur_corr = NR_corr[Nids[rem]-1][3]
            NR_corr[Nids[rem]-1][0].append(t)
            NR_corr[Nids[rem]-1][1].append(d)
            NR_corr[Nids[rem]-1][2].append(cur_corr + alpha*d)
            NR_corr[Nids[rem]-1][3] += alpha*d
        
        diff[(mp == rem).any(axis=1)] = np.nan
        
sta = 473;
x = np.array(NR_corr[sta-1][0])
y = np.array(NR_corr[sta-1][1])

def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind,
              np.take(x, ind), np.take(y, ind))

plt.close('all')
fig = plt.figure()
plt.scatter(x, y, picker=True)
fig.canvas.mpl_connect('pick_event', onpick3)
        

for i, row in enumerate(NR_corr):
    print(i+1)
    print(np.nanmean(row[1]))




