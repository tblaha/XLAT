# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:17:30 2020

@author: Till
"""

from .helper import C0, SP2CART

import numpy as np
import numpy.linalg as la


class Station_corrector():
    def __init__(self, TRA, NR, alpha):
        self.alpha = alpha

        self.mps = list()
        for n in range(max(TRA['M']) + 1):
            mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
            self.mps.append(mp)

        self.NRnp = np.zeros([len(NR), 3])
        for index, row in NR.iterrows():
            self.NRnp[index - 1, :] = SP2CART(
                NR.loc[index, ['lat', 'long', 'geoAlt']].to_numpy()
                )

        self.NR_corr = [list([[], [], [], 0]) for _ in range(len(NR))]

    def AbsoluteSync(self):
        clockErrors = np.array(
            [self.NR_corr[i][3] for i in range(len(self.NR_corr))]
            )
        correct = - np.median(clockErrors[abs(clockErrors) > 0])

        if np.isnan(correct):
            correct = 0

        for i in range(len(self.NR_corr)):
            self.NR_corr[i][3] += correct

    def RelativeSync(self, row, idx):
        M = row.iat[6]
        if M < 3:
            return

        mp = self.mps[M]
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
            var = np.zeros(M)

            for i in range(M):
                meas_i = (mp == i).any(axis=1)
                sign_i = (mp == i)[:, 0].astype(int) \
                            - (mp == i)[:, 1].astype(int)
                x = diff[meas_i] * sign_i[meas_i]
                var[i] = np.nanvar(x)**0.5
                med[i] = np.nanmedian(x)

            if np.nanmin(np.abs(var / med)) > 1e-1:
                break

            rem = np.nanargmin(np.abs(var / med))

            diffidx = np.where((mp == rem).any(axis=1)
                               & (~np.isnan(diff))
                               )[0][-1]

            cond = mp[diffidx, 0] == rem
            flip = 1 if cond else -1

            d = med[rem]

            if abs(d) < 5e4:
                cur_corr = self.NR_corr[Nids[rem]-1][3] + self.alpha * d
                self.NR_corr[Nids[rem]-1][0].append(t)
                self.NR_corr[Nids[rem]-1][1].append(d)
                self.NR_corr[Nids[rem]-1][2].append(cur_corr)
                self.NR_corr[Nids[rem]-1][3] = cur_corr

            diff[(mp == rem).any(axis=1)] = np.nan
            
    def ReconstructClocks(self, tserver, n):
        NR_idx = n - 1

        for i in NR_idx:
            for j, t in enumerate(self.NR_corr[i][0]):
                if t > tserver:
                    self.NR_corr[i][3] = self.NR_corr[i][2][j]
                    print(self.NR_corr[i][2][j])
                    break

