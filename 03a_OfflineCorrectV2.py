# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:45:59 2020

@author: Till
"""

import MLATlib as lib

from tqdm import tqdm

import numpy as np


# ### import and pre-process
use_pickle = True
use_file = 4  # -1 --> competition; 1 through 7 --> training

# MR, NR, SR = lib.read.importData(use_pickle, use_file)
# print("Finished importing data\n")


# use separate SR file for validation
# or
# select random data points with GT from MR set
np.random.seed(2)
use_SR = False
K = 10000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation

# TRA, VAL = lib.read.segmentDataByAC(MR, K=K, p=p_vali)
# TRA, VAL = lib.read.segmentData(MR, use_SR, SR=SR, K=K, p=p_vali)

TRA_GT = TRA.loc[~np.isnan(TRA['lat'])]

alpha = 0.2
NR_c = lib.sync.NR_corrector(TRA, NR, alpha)

for index, row in tqdm(TRA_GT.iterrows()):
    NR_c.RelativeSync(row, index)

lib.plot.StationErrorPlot(NR_c)
