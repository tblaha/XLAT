# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:40:56 2020

@author: Till
"""


import numpy as np
import numpy.linalg as la
from constants import *


def delta(n, x0):
    # 
    
    D  = np.array([(x0[0] - n[0]), (x0[1] - n[1]), (x0[2] - n[2])])
    nD = la.norm(D)
    
    d = np.array([ D[0], D[1], D[2] ]) / nD
    
    return d


def Jac(N, x0):
    # N is nx3 
    # 1, 2, ..., n
    # 12, 13, 14, 15, 23, 24, 25, 34, 35, 45...
    
    n = np.size(N, 0)
    
    ## generate mapping of distance vectors
    dim = int(n*(n-1)/2)
    mapping = np.zeros([dim, 2])
    
    idx = 0
    for i in np.arange(1,n):
        for j in np.arange(i+1,n+1):
            mapping[idx, :] = np.array([i, j])
            idx = idx + 1
    
    ## calculate row vectors and assemble matrix
    J = np.matrix( np.zeros([dim, 3]) )
    for i in np.arange(dim):
        J[i, :] = delta( N[int(mapping[i,1]-1), :], x0 ) \
        - delta( N[int(mapping[i,0]-1), :], x0 )
    
    return J, mapping


def iterx(N, T, xn):
    global C0
    
    J,mp = Jac(N, xn)
    
    delx = la.pinv(J) @ (T*C0 - (J @ xn).T)
    #print(delx)
    
    xnplus1 = np.array(xn + delx.T) # should be zero but isn't
    
    return xnplus1[0]
    

def NLLS_MLAT(MR, NR, idx):
    global X,Y,Z
    
    tmp = np.array(MR[MR.id == idx].n)
    stations = tmp[0]
    n = np.size(stations)
    
    lats  = np.array(NR[np.in1d(NR.n, stations)].iloc[:,1])
    longs = np.array(NR[np.in1d(NR.n, stations)].iloc[:,2])
    geoh  = np.array(NR[np.in1d(NR.n, stations)].iloc[:,3])
    
    N = np.array([ X(lats, longs, geoh), Y(lats, longs, geoh), Z(lats, longs, geoh) ]).T
    
    tmp = np.array(MR[MR.id == idx].ns)
    secs = tmp[0]*1e-9
    
    dim = int(n*(n-1)/2)
    T   = np.zeros([dim, 1])
    idx = 0
    for i in np.arange(1,n):
        for j in np.arange(i+1,n+1):
            T[idx, :] = secs[j-1] - secs[i-1]
            idx = idx + 1
    
    
    x0 = np.sqrt(3)/3*6371e3*np.array([1,1,1]) # centre of the earth
    itermax = 5 # max iterations
    
    it = 0
    xn = x0
    while it < itermax:
        xn = iterx(N,T,xn)
        print(xn)
        
        it = it + 1
    
    
    
#NLLS_MLAT(MR,NR,1439661)
#NLLS_MLAT(MR,NR,9999999)
    