# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:58:24 2020

@author: Till
"""

import MLATlib as lib
from MLATlib.helper import R0, X, Y, Z, SP2CART, CART2SP

import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

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
np.random.seed(1)
use_SR = True
K = 100000  # how many data points to read and use for validation
p_vali = 0.05  # share of K used for validation

TRA, VAL = lib.read.segmentData(MR, use_SR, SR, K=K, p=p_vali)
# TRA, VAL = lib.read.segmentDataByAC(MR, use_SR, SR, K=K, p=p_vali)

##########################

plt.close("all")


"""
##########################
# Histograms
##########################

bins = np.arange(np.min(MR['M']), np.max(MR['M'])+2)-0.5

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)

fig.suptitle("Number of Stations per Measurement -- Histograms",
             fontsize=16
             )

ax = axs[0]
ax.hist(MR['M'],
        bins=bins,
        density=True
        )
ax.set_xticks(ticks=bins[:-1]+0.5)
ax.grid()
ax.set_title("MR set %d; n = %d" % (use_file, len(MR)) )
_ = ax.set_xlabel("Number of Stations -- M")
ax.set_ylabel("Share of Data")

ax = axs[1]
ax.hist(TRA['M'],
        bins=bins,
        density=True
        )
ax.grid()
ax.set_title("TRA set %d; n = %d" % (use_file, len(TRA)) )
ax.set_xlabel("Number of Stations -- M")

ax = axs[2]
ax.hist(MR.loc[SR.index, 'M'],
        bins=bins,
        density=True
        )
ax.grid()
ax.set_title("SR set %d; n = %d" % (use_file, len(SR)) )
ax.set_xlabel("Number of Stations -- M")
"""


"""
#######################
# time drift
#######################


MR_Ntime = TRA.copy()

n = [list(elem) for elem in MR_Ntime["n"]]
ns = [list(elem) for elem in MR_Ntime["ns"]]
t = MR_Ntime["t"] * 1e9

lol = len(NR)*[None]
for idx, __ in enumerate(lol):
    lol[idx] = [[None],[None]]

for ne, nse, te in zip(n, ns, t):
    for nee, nsee in zip(ne, nse):
        lol[nee][0].append(te)
        lol[nee][1].append(nsee - te)


fig = plt.figure()
for l in lol:
    plt.scatter(l[0], l[1], s=1, marker='o')

plt.grid()
"""

"""
# ###################
# Error histograms
# ###################

columns = 4

HI = SEL.copy()
HI.loc[HI['NormError'] == 0, "NormError"] = np.nan

bins = np.concatenate([np.logspace(-1, 6, 15)])

M_list = np.unique(HI['M'])

rows = int(np.ceil(len(M_list) / columns))

fig, axs = plt.subplots(rows, columns, sharey=False, sharex=True)
fig.suptitle("Number of Stations per Measurement -- Histograms -- 826a8f",
             fontsize=16
             )

hists = []

for idx, m in enumerate(M_list):
    # for idx, m in enumerate([2, 3]):
    ax = axs[int(np.floor(idx / columns)), idx % columns]
    n, b, __ = ax.hist(HI.loc[HI['M'] == m, 'NormError'],
                   bins=bins,
                   density=False
                   )
    hists.append([m, n, b])
    # ax.set_xticks(ticks=bins[:-1]+0.5)
    ax.set_xscale('log')
    ax.grid()
    ax.set_title("SEL set 4 -- M = %d" % m)
    __ = ax.set_xlabel("2D Error")
    ax.set_ylabel("Datapoints")

return hists, fig, axs
"""
