# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 20:24:39 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la
import numpy.random as rd
import scipy.linalg as sla

from MLATlib.helper import C0

rd.seed(2)




#%% Definitions

def eqLine(x_ac, S, t, t_ac, lineid):
    f = la.norm(x_ac - S[:, lineid], axis=1)
    r = t_ac * C0
    y = t[:, lineid] * C0
    
    return f, r, y


def eqMat(x_ac, S, t, t_ac):
    f = la.norm((x_ac - S.swapaxes(0, 1)).swapaxes(0, 1), axis=2)
    r = t_ac * C0
    y = t * C0
    
    return f, r, y


def JeqMat(x_ac, S):
    R = (x_ac - S.swapaxes(0, 1)).swapaxes(0, 1)
    Rnorm = np.broadcast_to(
        la.norm((x_ac - S.swapaxes(0, 1)).swapaxes(0, 1), axis=2
                ), (3, k, n)
        ).swapaxes(1, 2).T

    A = np.ones([S.shape[0], S.shape[1], 4])
    A[:, :, 0:3] = R / Rnorm
    
    return A




#%% ground truth parameters
k = 2


### ac position
x_ac_hat = np.array([[0, 0, 5000]]).repeat(k, axis=0)
mu_ac = np.array([0, 0, 0])
Sigma_ac = np.diag([25, 25, 25])**2


### station positions
n = 5
S_hat = np.broadcast_to(np.array([[-10e3, -10e3, 0],
                                  [-10e3, +10e3, 0],
                                  [+10e3, -10e3, 0],
                                  [+10e3, +10e3, 0],
                                  [+0, +0, 0],
                                  ]),
                        (k, n, 3)
                        )

mu_st = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  ])
sigma_st = np.diag(np.array([1e-3, 1e-3, 1e-3]))**2
Sigma_st = np.array([sigma_st for i in range(n)])


### station-ac vectors
R_hat = la.norm((x_ac_hat - S_hat.swapaxes(0, 1)).swapaxes(0, 1), axis=2)
mu_cl = np.array([0, 0, 0, 500, 0])  # clock drift
Sigma_cl = np.diag([25, 25, 25, 25, 25])**2


### ToA measurement
t_hat = R_hat / C0




#%% Generate Measurements


### ac measurements
x_ac = x_ac_hat + rd.multivariate_normal(mu_ac, Sigma_ac, k)


### station measurements

# station location uncertainty
S = S_hat + (np.array([rd.multivariate_normal(mu_st[i], Sigma_st[i], k)
                      for i in range(n)
                      ]).swapaxes(0, 1)
             )


### station-ac vectors measurements

# station location uncertainty
R_x_hat = la.norm((x_ac_hat - S.swapaxes(0, 1)).swapaxes(0, 1), axis=2)

# x_ac and station location uncertainty
R = la.norm(x_ac - S.swapaxes(0, 1), axis=2
            ).swapaxes(0, 1)


### time measurements

# GT with clock uncertainty
t = (R_hat + rd.multivariate_normal(mu_cl, Sigma_cl, k)) / C0




#%% estimate mu_cl


A = np.ones(n)
W = np.diag(np.ones(n)/2)
fn, rn, yn = eqMat(x_ac, S, t, 0)

rac = np.zeros(k)

for j in range(k):
    for i in range(1):
        
        y = sla.sqrtm(W) @ (yn[j] - fn[j])
        
        rac[j] = 1/(A.T @ sla.sqrtm(W) @ A) * A.T @ y
        
        yres = sla.sqrtm(W) @ A*rac[j] - y
        
        w = np.diag(W) * 1/yres**2
        W = np.diag(w/la.norm(w))

tac = rac / C0

mu_cl_est = R[0] - (t[0] * C0 - rac[0])




#%% estimate mu_ac

DeltaX = 100*np.ones(n)
x_ac_est = np.zeros_like(x_ac)
x_ac_est[:, :] = x_ac
i = 0

for j in range(1):
    while i < 20 and la.norm(DeltaX) > 1e-4:
        
        # fn, rn, yn = eqMat(x_ac_est, S, t + mu_cl_est/C0, tac)
        # J = JeqMat(x_ac_est, S)
        
        fn, rn, yn = eqMat(x_ac_est, S, t, tac)
        J = JeqMat(x_ac_est, S)
        
        DeltaX = la.inv(J[j].T @ sla.sqrtm(W) @ J[j]) @ J[j].T @ sla.sqrtm(W)\
            @ (yn[j] - fn[j] - rn[j]).T
        
        print(DeltaX)
        
        x_ac_est[j] += DeltaX[0:3]
        tac[j] += DeltaX[3] / C0
        
        i += 1

print()
print(x_ac_est)



