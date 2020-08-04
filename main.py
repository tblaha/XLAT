# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""


#%% modules

import MLATlib as lib  # the main libraries that do the heavy lifting
from MLATlib.helper import (
    CART2SP,
    SP2CART,  # convert geodetic data [lat, long, alt] to ECEF [x, y, z]
    C0  # C0: vacuum speed of light
    )

import numpy as np
import pandas as pd

import pickle

from tqdm import tqdm  # awesome progress bar for iterative loops


#%% import

use_pickle = False  # False --> use unmodified .csv (slow)
use_file = 7  # -1 --> competition; 1 through 7 --> training

(Measurements,  # the huge 2M lines csv as pandas DF
 Stations,  # the stations with their location and so on
 Results  # the Results (bunch of NaNs for the competition set)
 ) = lib.read.importData(use_pickle, use_file)


#%% segment data

# True: use Results file for validation --> TRA = Measurements; VAL = Results
# ---or---
# False --> select random data with ADS-B ground truth from measurements set
use_Results = True

(TRA,  # training (also contains VAL set points, but NaNs instead of locations)
 VAL,  # validation
 ) = lib.read.segmentData(Measurements, use_Results, Results)


#%% insert pathological fakes for debugging
"""
<<< Removed >>>
"""


#%% investigate single measurment with plots

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
# seek_id = 503201  # best fit
# seek_id = 1823621  # 2 close stations mess it up
seek_id = 1869529
# seek_id = 1809908

# seek_id = idx_fake_planes[0]

NR_c_sp = lib.sync.Station_corrector(TRA, Stations, 0.1)



 # ### preprocess stations and measurements
# get number of stations
n = TRA.loc[seek_id, 'M']

# get station locations and convert to cartesian
stations = np.array(TRA.loc[seek_id, 'n'])
N = SP2CART(Stations.loc[stations, ['lat', 'long', 'geoAlt']].to_numpy())

# ### get unix time stamps of stations
Rs_corr = np.array([NR_c_sp.NR_corr[i - 1][3] for i in stations])
Rs = np.array(TRA.loc[seek_id, 'ns']) * 1e-9 * C0 + Rs_corr  # meters

# baro radius
h_baro = TRA.loc[seek_id, 'baroAlt']  # meters

# gen measurements
mp, RD, R, Rn, RD_sc = lib.ml.GenMeasurements(N, n, Rs)

# determine problem size
dim = len(mp)

# ### calculate quadratic form
A, V, D, b, singularity = lib.ml.getHyperbolic(N, mp, dim, RD, R, Rn)

# ### generate x0
x0 = lib.ml.genx0(N, mp, RD_sc, h_baro)

# build diagnostic struct
inDict = {'A': A, 'b': b, 'V': V, 'D': D, 'dim': dim, 'RD': RD, 'xn': x0,
          'fun': lambda x, m: lib.ml.FJsq(x, A, b, dim, V, RD, Rn, mode=m),
          'xlist': np.zeros([2, 3]), 'ecode': 0, 'mp': mp, 'Rn': Rn, 'sol': {}}


# start MLAT calculations
#x_sph, inDict = lib.ml.Pandas_Wrapper(TRA, Stations, seek_id, NR_c_sp, solmode='2d')

pp = lib.plot.HyperPlot(TRA, VAL, Stations, seek_id, CART2SP(x0), inDict, SQfield=False)

#print(la.norm(SP2CART(x_sph) - SP2CART(plane_sph[0])))


#%% initialize

###### Pandas Dataframes ######
# initialise solution dataframe to store the final answers
SOL = VAL.copy()
SOL[["lat", "long", "geoAlt"]] = np.nan  # needed for Training sets,
                                         # redundant for Competition set which
                                         # is all-NaN anyway

# add column to store the MLATtime with the training data
TRA['MLATtime'] = np.nan

# column to indicate if a value was succesfully computed and used
TRA['MLAT'] = False


###### Clock Corrections ######
alpha = 2e-1  # Filter constant for discrete IIR clock offset filter

# clock corrector class (unit of corrections --> meters!)
Sta_corr = lib.sync.Station_corrector(TRA, Stations, alpha)


###### Temporary Numpy arrays ######
# (lat, long, alt)-np array for the computed solution
# (for faster individual write access than the pandas SOL dataframe)
xn_sph_np = np.zeros([len(SOL), 3])
xn_sph_np[:, :] = np.nan

# ground truth training set as np array (for faster individual read access)
x_GT = SP2CART(VAL.to_numpy())

# array to hold the function residual when applying the MLAT objective
# function to the ground truth postition
# --> will remain NaN for the competition set
fval_GT = np.zeros(len(SOL))
fval_GT[:] = np.nan


#%% Start iterating

npi = 0  # numpy index for fval_GT, x_GT and xn_sph_np
for idx, row in tqdm(TRA.iterrows(), total=len(TRA)):
    if idx not in SOL.index:
        """Not part of Validation set, so we have ground thruth ADS-B data
        --> do a relative clock sync"""
        Sta_corr.RelativeSync(row, idx)

    else:
        """Part of Results set: attempt to compute location"""
        try:
            # don't compute anything in the first 6 minutes because the clocks
            # are not correct yet
            # TODO: do this properly with row['t']
            assert(idx > 6*60/3600*len(TRA))

            """attempt MLAT calculation"""
            (xn_sph_np[npi],  # spherical location estimate [lat, long, alt]
             DebugDict,       # additional debugging information
             ) = lib.ml.Pandas_Wrapper(
                     TRA,           # the measurements
                     Stations,      # the stations
                     idx,           # the index for which to solve
                     Sta_corr,      # station clock corrector
                     solmode='2d',  # solve 2d using baroAlt
                     )

            if len(DebugDict):
                # calculation was successful
                # --> compute function residual when supplying the ground
                #     truth location data
                fval_GT[npi] = DebugDict['fun'](x_GT[npi], 0)

        except (lib.ml.MLATError, AssertionError):
            # do nothing here --> location estimate will be NaN, status code
            #                     in the TRA dataframe will be >0 and all of
            #                     this will be handled later
            pass

        finally:
            # no matter the result of the try-except above, do this:

            # increment the numpy index
            npi += 1

            # estimate the syncronised MLAT receiver time and write it to the
            # measurement for later use
            # --> \sum_i^M(receiver_time_measurements + clock_corrections) / M
            TRA.at[idx, 'MLATtime'] = np.mean(
                np.array([TRA.loc[idx, 'ns']])  # time measurements
                + np.array([Sta_corr.NR_corr[TRA.loc[idx, 'n'][i] - 1][3] \
                            for i in range(len(TRA.loc[idx, 'ns']))
                            ]) / C0 * 1e9  # clock corrections in nano-seconds
                )


###### write the numpy arrays to the pandas dataframes ######
SOL[["lat", "long", "geoAlt"]] = xn_sph_np
TRA.loc[SOL.index, ['lat', 'long', 'geoAlt']] = xn_sph_np


###### write down intermediate results ######
# TRA.to_pickle("./TRA_7_da421b_.pkl")
# SOL.to_pickle("./SOL_7_da421b_.pkl")
# with open('NRc_fvalGT_da421b_7.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([NR_c, fval_GT], f)


#%% Prune to trustworthy data

def prune(SEL):
    # returns a boolean array estimate if a position estimate should be
    # trusted or not based on:
    #   - its objective function residual fval
    #   - the amount of TDOA measurements used
    #     (M choose 2 = M*(M-1)/2 or less)
    discard = SEL['fval']\
        > (10**(np.clip(7 + 0.5 * np.clip(SEL['dim'] - 6,
                                          0,
                                          1e4)**0.6,
                        5,
                        11)
                ))

    return discard


def PruneResults(TRA, SOL, prunefun=None):

    TRAt = TRA.copy()
    SOLt = SOL.copy()

    if prunefun is None:
        prunefun = lambda TRAt: TRAt['fval'] > 1e8

    idx_prune = TRAt.loc[prunefun(TRAt)].index
    TRAt.loc[idx_prune, ["lat", "long", "geoAlt"]] = np.nan
    TRAt.loc[idx_prune, "MLAT_status"] = 4
    SOLt.loc[idx_prune, ["lat", "long", "geoAlt"]] = np.nan

    return TRAt, SOLt


# do the pruning
TRA_pruned, SOL_pruned = PruneResults(TRA, SOL, prunefun=prune)

# indicate those MLAT results that were actually used
TRA_pruned.loc[TRA_pruned['MLAT_status'] == 0, 'MLAT'] = True


#%% filtering and interpolation

TRA_filtered = TRA_pruned.copy()
SOL_filtered = SOL_pruned.copy()

# column to hold the curvatore score of the computed points after filtering
TRA_filtered['score'] = np.nan

# iterate over the measurements of each aircraft
acs = np.unique(TRA_pruned.loc[SOL_pruned.index, 'ac'])
for ac in tqdm(acs):
    # create filter class instance for
    ac_filter = lib.filt.aircraft(TRA_pruned, SOL_pruned, ac)

    # interpolate with adaptive node-selection
    ac_filter.Interp(usepnts='adaptive')

    # insert the solutions into the dataframes
    TRA_filtered.loc[ac_filter.ids] = ac_filter.TRAac
    SOL_filtered.loc[ac_filter.ids] = ac_filter.SOLac


#%% keep only best 50% by curvature score --> this might be a hoax...

# estimate the percentile of successful estimates (either by direct MLAT or by
# interpolation) to keep at least the best 0.5*len(SOL) solutions
coverageEst = sum(~np.isnan(TRA_filtered['score'])) / len(SOL_filtered)
keepPercentile = min(0.50 / coverageEst, 1)

# determine which indexes to lose to get low curvature of the measurements
loseIndex = TRA_filtered.index[
    TRA_filtered['score'] > TRA_filtered['score'].quantile(keepPercentile)
    ]

TRA_top50 = TRA_filtered.copy()
SOL_top50 = SOL_filtered.copy()

TRA_top50.loc[loseIndex, ['lat', 'long', 'geoAlt']] = np.nan
SOL_top50.loc[loseIndex, ['lat', 'long', 'geoAlt']] = np.nan


#%% Print final accuracy

RMSE, cov, nv = lib.out.twoErrorCalc(SOL_top50, VAL, RMSEnorm=2)

print(RMSE)
print(cov*100)


#%% write solution

lib.out.writeSolutions("../Comp1_da421b_0.2_37_1.0.csv", SOL_top50)
# lib.out.writeSolutions("../Train7_da421b.csv", SOL_top50)


#%% append some GT data and sort the final data frames
"""only works for Training sets, of course"""

TRA_top50.loc[SOL_top50.index, 'fval_GT'] = fval_GT
TRA_top50.loc[VAL.index, "NormError"] = nv
SEL = TRA_top50.loc[~np.isnan(TRA_top50.NormError)]\
    .sort_values(by="NormError", ascending=True)


# !!! NO MORE ACCURACY-IMPROVING CODE AFTER HERE!!! It would be "cheating" and
# wouldn't work on the competition set


#%% plotting

# Error plots --> only works with Training sets, of course
# lib.plot.ErrorCovariance(SEL)
# lib.plot.ErrorHist(SEL)

# Track plots
pp = lib.plot.PlanePlot()
pp.addTrack(TRA_top50, acs, z=VAL, color='orange')


#%% EOF
