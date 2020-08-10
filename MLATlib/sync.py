# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:17:30 2020

@author: Till
"""

from .helper import (
    SP2CART,  # convert geodetic data [lat, long, alt] to ECEF [x, y, z]
    C0  # C0: vacuum speed of light
    )

import numpy as np
import numpy.linalg as la


class Station_corrector():
    """
    Provides clock correction in meters (self.NR_corr[i][3]) by using ADS-B 
    measurements to update the correction.
    
    """
    def __init__(self, TRA, NR, alpha):
        
        # discrete filter constant
        self.alpha = alpha

        # generate all possible maps (TDOA measurement id --> pair of stations)
        self.maps = list()
        for n in range(max(TRA['M']) + 1):
            mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
            self.maps.append(mp)

        # generate a numpy array to hold all of the station's ECEF locations
        # (for faster access)
        self.NRnp = SP2CART(NR[['lat', 'long', 'geoAlt']].to_numpy())

        # initialize the Station clock correction list
        # NR_corr[n - 1][0] --> list of server times at time of correction
        # NR_corr[n - 1][1] --> list of drifts of correction wrt to previous
        # NR_corr[n - 1][2] --> list of total clock correction
        # NR_corr[n - 1][3] --> current clock correction
        self.NR_corr = [list([[], [], [], 0]) for _ in range(len(NR))]
        
        self.ac_disc = []

    def AbsoluteSync(self):
        """ this is unused """
        
        # the relative sync is prone to "absolute drifts" after some time 
        # (hours), since it doesn't matter if the relative reciever times are 
        # meaningful in the absolute sense --> gotta do an Absolute Sync every
        # once in a 2hile
        # BUT:
        # TODO: not like this... this, for some reason this induces 
        #       oscillations
        clockErrors = np.array(
            [self.NR_corr[i][3] for i in range(len(self.NR_corr))]
            )
        correct = - np.median(clockErrors[abs(clockErrors) > 0])

        if np.isnan(correct):
            correct = 0

        for i in range(len(self.NR_corr)):
            self.NR_corr[i][3] += correct

    def RelativeSync(self, row, idx):
        # correct clocks by comparing ground truth TDOA (or rather Range 
        # Differences) to the measurements and trying to infere which station
        # is off by looking at variances and means of the TDOA errors
        
        M = row.iat[6]
        if M < 3:
            return
        
        if row['ac'] in self.ac_disc:
            return

        mp = self.maps[M]
        Nids = np.array(row.iat[8])
        N = self.NRnp[Nids - 1]
        t = row.iat[0]

        x_sph_GT = row.iloc[[2, 3, 4]].to_numpy()
        x_GT = SP2CART(x_sph_GT)

        Rs_GT = la.norm(x_GT - N, axis=1)
        RD_GT = (- Rs_GT[mp[:, 1]] + Rs_GT[mp[:, 0]])

        Rs_corr = np.array([self.NR_corr[i - 1][3] for i in Nids])

        Rs = np.array(row.iat[9]) * 1e-9 * C0 + Rs_corr
        RD = (- Rs[mp[:, 1]] + Rs[mp[:, 0]])

        diff = RD_GT - RD

        while sum((~np.isnan(diff)).astype(int)) > 1:
            med = np.zeros(M)
            std = np.zeros(M)

            for i in range(M):
                meas_i = (mp == i).any(axis=1)
                sign_i = (mp == i)[:, 0].astype(int) \
                            - (mp == i)[:, 1].astype(int)
                x = diff[meas_i] * sign_i[meas_i]
                std[i] = np.nanvar(x)**0.5
                med[i] = np.nanmedian(x)

            score = np.abs(std)

            if (np.nanmin(score) > 80) or (np.nanmin(std/abs(med)**0.5) > 0.6):
                self.ac_disc.append(row['ac'])
                break

            rem = np.nanargmin(score)

            d = med[rem]

            if abs(d) < 5e4:
                cur_corr = self.NR_corr[Nids[rem] - 1][3] + self.alpha * d
                self.NR_corr[Nids[rem] - 1][0].append(t)
                self.NR_corr[Nids[rem] - 1][1].append(d)
                self.NR_corr[Nids[rem] - 1][2].append(cur_corr)
                self.NR_corr[Nids[rem] - 1][3] = cur_corr

            diff[(mp == rem).any(axis=1)] = np.nan
            
    def ReconstructClocks(self, tserver, n):
        """ not used in the final code, just for debugging after the fact"""
        
        NR_idx = n - 1

        if np.isscalar(n):
            a = np.zeros(0)
        else:
            a = np.zeros(len(n))
        
        for i in NR_idx:
            for j, t in enumerate(self.NR_corr[i][0]):
                if t > tserver:
                    self.NR_corr[i][3] = self.NR_corr[i][2][j]
                    a[i] = self.NR_corr[i][2][j]
                    break
        
        return a

