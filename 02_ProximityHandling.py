# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:59:29 2020

@author: Till
"""


import MLATlib as lib
from MLATlib.helper import SP2CART
from MLATlib import MLAT_hyp2 as ml

import os
import time
from tqdm import tqdm

import numpy as np
import numpy.linalg as la
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib import pyplot as plt


"""
# ### import and pre-process
use_pickle = False
use_file = 4  # -1 --> competition; 1 through 7 --> training

MR, NR, SR = lib.read.importData(use_pickle, use_file)
# print("Finished importing data\n")


# use separate SR file for validation
# or
# select random data points with GT from MR set
np.random.seed(2)
use_SR = True
K = 20000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation


TRA, VAL = lib.read.segmentData(MR, use_SR, SR, K=K, p=p_vali)
# TRA, VAL = lib.read.segmentDataByAC(MR, K, p_vali)
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
seek_id = 3276  # comp set -1

# start MLAT calculations
x_sph, inDict = ml.NLLS_MLAT(MR, NR, seek_id)
x_sph_GT = MR.loc[seek_id, ['lat', 'long', 'geoAlt']]

print(la.norm(SP2CART(x_sph[0], x_sph[1], x_sph[2])
              - SP2CART(x_sph_GT[0], x_sph_GT[1], x_sph_GT[2])
              ))

print(len(inDict['xlist']))

pp = lib.plot.HyperPlot(MR, SR, NR, seek_id, x_sph, inDict, SQfield=True)




