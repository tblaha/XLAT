# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 00:18:40 2020

@author: Till
"""
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

import pickle

if False: 
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    
    
#%% init

def prune(SEL, lam):
    return SEL['fval'] > lam*(10**(np.clip(7 + 0.5*np.clip(SEL['dim'] - 6, 0, 1e4)**0.6, 5, 11)))
    # return (~SEL['Interior'] & SEL['fval'] > 0) | (SEL['dim'] < 3) | (SEL['fval'] > 1e9) # cheat!!!
    # return SEL['fval'] > 1e8


# eTHs, lams = np.meshgrid(np.array([5, 10, 20]),
#                          np.array([0.5, 0.8]))

eTHs, lams = np.meshgrid(np.array([40]),
                          np.array([0.8, 1]))


for i, __ in enumerate(eTHs[0]):
    for j, __ in enumerate(lams[:, 0]):
        eTH = eTHs[j, i]
        lam = lams[j, i]
        
        
        #%% pruning
        
        TRA_temp, SOL_temp = lib.ml.PruneResults(TRA, SOL, prunefun=lambda x: prune(x, lam))
        
        
        #%% filtering
        
        TRA_temp['MLAT'] = False
        TRA_temp.loc[TRA_temp['MLAT_status'] == 0, 'MLAT'] = True
        
        TRA2 = TRA_temp.copy()
        TRA2['score'] = np.nan
        
        SOL2 = SOL_temp.copy()
        acs = np.unique(TRA2.loc[SOL2.index, 'ac'])
        for ac in tqdm(acs):
            aco = lib.filt.aircraft(TRA_temp, SOL_temp, ac)
            aco.eTH = eTH
            aco.Interp(usepnts='adaptive')
            SOL2.loc[aco.ids] = aco.SOLac
            TRA2.loc[aco.ids] = aco.TRAac
            
            
        #%% keep only best 50% by score

        covEst = len(TRA2[~np.isnan(TRA2['score'])]) / len(SOL2)
        keepPercentile = min(0.50 / covEst, 1)
        loseIndex = TRA2.index[TRA2['score'] > TRA2['score'].quantile(keepPercentile)]
        
        TRA3 = TRA2.copy()
        SOL3 = SOL2.copy()
        
        TRA3.loc[loseIndex, ['lat', 'long', 'geoAlt']] = np.nan
        SOL3.loc[loseIndex, ['lat', 'long', 'geoAlt']] = np.nan
        
        
        #%% print final accuracy
        
        RMSE, cov, nv = lib.out.twoErrorCalc(SOL3, VAL, RMSEnorm=2)

        print(RMSE)
        print(cov*100)
        
        
        #%% write
        
        lib.out.writeSolutions("../Comp1_da421b_0.2_"+str(eTH)+"_"+str(lam)+".csv", SOL3)
