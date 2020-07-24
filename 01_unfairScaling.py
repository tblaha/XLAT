# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:40:30 2020

@author: Till
"""

import MLATlib as lib

import numpy as np
import numpy.linalg as la

from matplotlib import pyplot as plt


N = np.array([[0, 0, -10], [0, 0, 10]])
xGT = np.array([0, 0, 5])

mp = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
RD2 = np.array([0, 0.125, 0.25, 10])
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
A, V, D, b, singularity = lib.ml.getHyperbolic(N, mp, dim, RD, R, Rn)

scaling = np.zeros(dim)
scaling[RD_sc == 0] = 1
scaling[RD_sc != 0] = RD_sc[RD_sc != 0]**2

# get Errors
Res = np.zeros([4, 1001])
zGT = np.linspace(-3, 3, 1001)
for idx, z in enumerate(zGT):
    Res[:, idx] = lib.ml.FJsq(np.array([0, 0, z]),
                              A, b, dim, V, RD, Rn, mode=-1,
                              singularity=singularity)

# print(F * scaling)

plt.plot(zGT, Res.T**2)
plt.grid()
plt.legend(["0", "0.125", "0.25", "-0.5"])

