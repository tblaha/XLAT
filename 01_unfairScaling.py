# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:40:30 2020

@author: Till
"""

from MLATlib import MLAT_hyp2 as ml

import numpy as np
import numpy.linalg as la

from matplotlib import pyplot as plt


N = np.array([[0, 0, -1], [0, 0, 1]])
xGT = np.array([0, 0, 5])

mp = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
RD2 = np.array([0, 0.125, 0.25, -0.5])
RD = RD2*2
dim = len(RD)

# range differences RD (equivalent to TDOA)
# RD = (- Rs[mp[:, 1]] + Rs[mp[:, 0]])  # meters

# vectors and ranges between stations
R = (N[mp[:, 1]] - N[mp[:, 0]])
Rn = la.norm(R, axis=1)

# Scaled measurements
RD_sc = RD/Rn

# get hyperbolas
A, V, D, b, singularity = ml.getHyperbolic(N, mp, dim, RD, R, Rn)

scaling = np.zeros(dim)
scaling[RD_sc == 0] = 1
scaling[RD_sc != 0] = RD_sc[RD_sc != 0]**2

# get Errors
Res = np.zeros([4, 1001])
zGT = np.linspace(-2, 2, 1001)
for idxz in range(1001):
    Res[:, idxz] = ml.FJsq(np.array([0, 0.5, zGT[idxz]]),
                           A, b, dim, V, RD, Rn, mode=-1,
                           singularity=singularity)

plt.close('all')
plt.figure()
plt.plot(zGT, Res.T)
plt.legend(["0", "0.125", "0.25", "-0.5"])
plt.grid()



# get Errors
k = 51
Res = np.zeros([4, k, k])
Del = np.zeros([3, 4, k, k])
Del[:, :, :, :] = np.nan
yGT, zGT = np.meshgrid(np.linspace(-5, 5, k), np.linspace(-1, 1, k))
for idxy in range(k):
    for idxz in range(k):
        Res[:, idxy, idxz] = ml.FJsq(np.array([0, yGT[idxy, idxz], zGT[idxy, idxz]]),
                                     A, b, dim, V, RD, Rn, mode=-1,
                                     singularity=singularity)
        if (not idxy%2) & (not idxz%2):
            Del[:, :, idxy, idxz] = ml.FJsq(np.array([0, yGT[idxy, idxz], zGT[idxy, idxz]]),
                                            A, b, dim, V, RD, Rn, mode=-2,
                                            singularity=singularity).T


plt.figure()
cs = plt.contour(yGT, zGT, Res[1])
plt.clabel(cs)
plt.quiver(yGT, zGT, Del[1, 1, :, :], Del[2, 1, :, :],
           headwidth=3, headlength=2, headaxislength=1.5)
plt.grid()
ax = plt.gca()
ax.set_aspect('equal')

