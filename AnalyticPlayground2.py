# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:55:25 2020

@author: Till
"""

import numpy as np
from mayavi import mlab
import MLAT_hyp as mlh
import time
import numpy.linalg as la
from matplotlib import pyplot as plt

"""
np.random.seed(0)
N = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]])
# N  = np.array([[0,0,1], [0,0,-1], [0,1,0], [1,0,0], [0,-1,0]])
n = len(N)
x_GT = np.array([0.5, 0, 0])  # + 0.1*np.random.random(3) * np.array([1,1,1])
Rs = la.norm(x_GT - N, axis=1)  # + 0.001*np.random.random(n)
mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
"""



x_sph, inDict = mlh.MLAT(N, n, Rs, rho_baro=r_baro)
# print(xn)
# print(x_GT)

Ffun = lambda x: mlh.FJ(x,
       inDict['A'], inDict['b'], inDict['dim'], inDict['V'], inDict['RDsi'],
       mode=0)
    
Jfun = lambda x: mlh.FJ(x,
       inDict['A'], inDict['b'], inDict['dim'], inDict['V'], inDict['RDsi'],
       mode=1)

dim = inDict['dim']


##
n_vals = 100
x, y = np.meshgrid(np.linspace(-3, 3, n_vals),
                   np.linspace(-3, 3, n_vals))

F = np.zeros([dim, n_vals, n_vals])
Fsq = np.zeros([dim, n_vals, n_vals])
J = np.zeros([dim, 3, n_vals, n_vals])

for i in range(n_vals):
    for j in range(n_vals):
        xvec = np.array([x[i, j], y[i, j], 0])
        F[:, i, j] = (Ffun(xvec))
        Fsq[:, i, j] = F[:, i, j]**2
        J[:, :, i, j] = Jfun(xvec)
        
K = np.arange(-20, 20.1, 0.5)


fig, ax = plt.subplots()
cs = ax.contour(x, y, F[0], K)
ax.clabel(cs, fontsize=10)
ax.grid()


"""
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, F[0], alpha=0.5)
cs = ax.contour(x, y, F[0], [0.05], linewidths=5)
"""


"""

## Itterazzione
n_vals = 40
x, y, z = np.mgrid[-3:3:n_vals*1j,\
                   -3:3:n_vals*1j,\
                   -3:3:n_vals*1j]

i = 0
j = 0
k = 0
F = np.zeros([dim, n_vals, n_vals, n_vals])
J = np.zeros([dim, 3, n_vals, n_vals, n_vals])
for i in np.arange(n_vals):
    for j in np.arange(n_vals):
        for k in np.arange(n_vals):
            xvec = np.array([x[i,0,0], y[0,j,0], z[0,0,k]])
            F[:, i,j,k] = Ffun(xvec)

n_vals = 20
xj, yj, zj = np.mgrid[-3:3:n_vals*1j,\
                      -3:3:n_vals*1j,\
                      -3:3:n_vals*1j]

i = 0
j = 0
k = 0
J = np.zeros([dim, 3, n_vals, n_vals, n_vals])
for i in np.arange(n_vals):
    for j in np.arange(n_vals):
        for k in np.arange(n_vals):
            xvec = np.array([xj[i,0,0], yj[0,j,0], zj[0,0,k]])
            J[:, :, i,j,k] = Jfun(xvec)

## plottare
mlab.close(all=True)

ck = [(0,0,1), (0,1,0), (1,0,0), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]
for idx in range(1):#range(dim):
    mlab.contour3d(x, y, z, F[idx, :,:,:], transparent=True, contours = [0], opacity=0.6, color = ck[idx])
    mlab.quiver3d(xj, yj, zj, J[idx, 0], J[idx, 1], J[idx, 2])
    
for idx in range(n):
    mlab.points3d(N[idx, 0], N[idx, 1], N[idx, 2], color = (1,1,1), scale_factor=1e-1)

mlab.points3d(x_GT[0], x_GT[1], x_GT[2], color = (0,1,0), scale_factor=1e-1)
mlab.points3d(xn[0], xn[1], xn[2], color = (1,0,0), scale_factor=1e-1)
"""

"""
mlab.plot3d(x2, y2, z2, color=(1,1,0), tube_radius=None)
mlab.plot3d(x3, y3, z3, color=(1,1,0), tube_radius=None)
mlab.points3d(p1[0], p1[1], p1[2], scale_mode='none', scale_factor=1e-1, color=(1,1,0))

    
mlab.points3d(x_GT[0],x_GT[1],x_GT[2], scale_mode='none', scale_factor=3e-1, color=(0,1,0))
mlab.points3d(P1[0],P1[1],P1[2], scale_mode='none', scale_factor=3e-1, color=(1,1,1))
mlab.points3d(P2[0],P2[1],P2[2], scale_mode='none', scale_factor=3e-1, color=(1,1,1))
mlab.points3d(b[0],b[1],b[2], scale_mode='none', scale_factor=3e-1, color=(0,0,1))
mlab.quiver3d(b[0]*np.ones([2]), b[1]*np.ones([2]), b[2]*np.ones([2]), V[0,1:],V[1,1:],V[2,1:])
mlab.quiver3d(0*np.ones([3]), 0*np.ones([3]), 0*np.ones([3]), [1,0,0], [0,1,0], [0,0,1], color=(0,0,0))
mlab.quiver3d(b[0], b[1], b[2], V[0,0],V[1,0],V[2,0], color=(0,1,1))
"""
