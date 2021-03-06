# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 07:29:08 2020

@author: Till
"""

import MLATlib as lib
import MLATlib.MLAT as ml
from MLATlib.helper import X, Y, Z, R0, C0, SP2CART

import sys

import numpy as np
import scipy.optimize as sciop

from mayavi import mlab


idx = MR.iloc[384350].id

### preprocess stations and measurements
    
# find stations
stations = np.concatenate(MR[MR.id == idx].n.to_numpy())
n = MR[MR.id == idx].M.iloc[0]

# convert station locations to cartesian
lats  = NR.set_index('n').loc[stations].reset_index(inplace=False).lat.to_numpy()
longs = NR.set_index('n').loc[stations].reset_index(inplace=False).long.to_numpy()
geoh  = NR.set_index('n').loc[stations].reset_index(inplace=False).geoAlt.to_numpy()

N = np.array([ X(lats, longs, geoh), Y(lats, longs, geoh), Z(lats, longs, geoh) ]).T

# find number of TDOA measurements available
dim = int(n*(n-1)/2)
if dim < 3:
    print("not enough stations")
    sys.exit()


# ground truth
plane_cart = SP2CART(MR[MR.id == idx].lat.iloc[0], MR[MR.id == idx ].long.iloc[0], MR[MR.id == idx ].geoAlt.iloc[0])
TOT_cheat = np.array([la.norm(N - plane_cart, axis=1)/C0]).T
        

# grab station TOA
secs = np.concatenate(MR[MR.id == idx].ns.to_numpy())*1e-9 # get nanoseconds into seconds
#secs[1] = secs[1] + 1.2048*1e-4

# pre alloc 
b         = np.zeros([dim+1, 1])
b_cheat   = np.zeros([dim+1, 1])
mut_dists = np.zeros([dim, 1])
mp        = np.zeros([dim, 2]) # Mapping matrix, maps a time difference 
                               # index to the indices of the subtrahend and minuend.

# iterate over the possible differences between 2 stations out of n
index = 0
for i in np.arange(1,n):
    for j in np.arange(i+1,n+1):
        b[index, :] = secs[j-1] - secs[i-1]
        b_cheat[index, :] = TOT_cheat[j-1] - TOT_cheat[i-1]
        mut_dists[index] = la.norm(N[j-1]-N[i-1])
        mp[index, :] = np.array([i, j])
        index = index + 1

# make mapping contain on ints (for indexing with it later)
mp = np.vectorize(int)(mp)

# add altitude (radius) target from baroAlt (light speed normalized)
b[-1]       = (MR[MR.id == idx].baroAlt + R0) / C0
b_cheat[-1] = (MR[MR.id == idx].geoAlt  + R0) / C0


# scaled TDOA ranges by distances between stations
mut_dists_sc = b[0:-1]*C0/mut_dists 

b = b_cheat
mut_dists_sc = b[0:-1]*C0/mut_dists 
active_pnts = np.in1d(np.arange(dim), np.argsort(abs(mut_dists_sc).T[0])[:2])


x0_idx = np.where(abs(mut_dists_sc) == np.min(abs(mut_dists_sc)))[0][0]
#x0_idx = np.where(abs(mut_dists_sc) == np.max(abs(mut_dists_sc)))[0][0]
#x0 = plane_cart + np.ones([3])*np.random.rand(1)*1e2
x0 = N[mp[x0_idx,0]-1,:] + (0.5-mut_dists_sc[x0_idx]/2) * (N[mp[x0_idx,1]-1,:] - N[mp[x0_idx,0]-1,:])
#x0 = np.array([4377406.13307837,  464053.99364404, 4632154.30923557])

llsq_active = np.concatenate( (active_pnts, [True]) )

# use scipy's LSQ solver based on Levenberg-Marqart with custom Jacobian
# only solve at active_points
sol = sciop.least_squares(\
    lambda x: np.array(lib.ml.fun(N, mp, x).T)[0][llsq_active] - b.T[0][llsq_active] * C0, \
    x0, \
    jac=lambda x: np.array(lib.ml.Jac(N, mp, x)[llsq_active]), \
    #method='dogbox', x_scale='jac', loss='linear', tr_solver='exact', \
    #method='dogbox', x_scale='jac', loss='soft_l1', f_scale=1e4, tr_solver='exact', \
    #method='dogbox', x_scale='jac', loss=lambda x: rho_alt(x, 1e0), f_scale=1e4, tr_solver='exact', \
    method='lm', x_scale='jac', \
    max_nfev=200, xtol=1e-8, gtol=1e-8, ftol=None, \
    verbose=2)
xn = sol.x
#print(sol.success)
#print(sol.message)

sol2 = sciop.minimize(lambda x: la.norm(lib.ml.fun(N, mp, x)[[0,1,2]] - b[[0,1,2]]*C0)**2, \
                      x0,\
                      method='SLSQP',\
                      jac=lambda x: lib.ml.fun(N, mp, x)[0:-1] / la.norm(lib.ml.fun(N, mp, x)[0:-1]),\
                      hess=lambda x: Jac(N, mp, x)[0:-1],\
                      constraints = sciop.NonlinearConstraint(\
                            lambda x: (lib.ml.fun(N, mp, x)[-1] - b[-1]*C0), 0, 0))#, jac=lambda x: lib.ml.Jac(N, mp, x)[-1]) )
xn = sol2.x


#sol3 = sciop.root(\
#    lambda x: np.array(lib.ml.fun(N, mp, x).T)[0][llsq_active] - b.T[0][llsq_active] * C0, \
#    x0, \
#    method='lm', \
#    jac=lambda x: np.array(lib.ml.Jac(N, mp, x)[llsq_active]), \
#    options = {'ftol':1e-4, 'xtol':1e-10} )

#xn = sol3.x


mlab.close(all = True)
mlab.figure()
mlab.points3d(x0[0], x0[1], x0[2], scale_mode='none', scale_factor=5e3, color=(0,1,1))
mlab.text(x0[0], x0[1],  "Init x0", z=x0[2], width=0.13)

mlab.points3d(xn[0], xn[1], xn[2], scale_mode='none', scale_factor=5e3, color=(0,1,0))
mlab.text(xn[0], xn[1],  "Localized A/C", z=xn[2], width=0.13)



#######
import time

NandPlane = np.concatenate((N, plane_cart[:,np.newaxis].T, xn[:, np.newaxis].T))
n_vals = 25
x, y, z = np.mgrid[np.min(NandPlane[:,0])-250:np.max(NandPlane[:,0])+250:n_vals*1j,\
                   np.min(NandPlane[:,1])-250:np.max(NandPlane[:,1])+250:n_vals*1j,\
                   np.min(NandPlane[:,2])-250:np.max(NandPlane[:,2])+250:n_vals*1j]

t = time.time()

i = 0
j = 0
k = 0
F = np.zeros([dim+1, n_vals, n_vals, n_vals])
fval = np.zeros([n_vals, n_vals, n_vals])
for i in np.arange(n_vals):
    for j in np.arange(n_vals):
        for k in np.arange(n_vals):
            F[:,i,j,k]  = lib.ml.fun(N, mp, np.array([x[i,0,0], y[0,j,0], z[0,0,k]])).T[0]
            fval[i,j,k] = la.norm(F[:,i,j,k].squeeze() - b.squeeze()*C0)**2
            

elapsed = time.time() - t


colors = [(0,0,1), (0,1,0), (1,0,0), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]
#hyp=mlab.figure()
for i in np.arange(dim+1):
    if llsq_active[i]:
        mlab.contour3d(x, y, z, F[i,:,:,:], contours = [b_cheat[i,0]*C0], color=colors[i], transparent=True, opacity=0.6)
        #mlab.contour3d(x, y, z, F[i,:,:,:], contours = [b[i,0]*C0], color=(1,1,1))

#mlab.contour3d(x, y, z, fval, contours = 4, color=(1,0,0), transparent=True, opacity=0.6)
    

for i in np.arange(n):
    mlab.points3d(N[i,0], N[i,1], N[i,2], scale_mode='none', scale_factor=5e3)
    mlab.text(N[i,0], N[i,1], "Node %d"%(i+1), z=N[i,2], width=0.13)

mlab.points3d(plane_cart[0], plane_cart[1], plane_cart[2], scale_mode='none', scale_factor=5e3, color=(0,0,1))
#mlab.show() # whatever, this crashes python

