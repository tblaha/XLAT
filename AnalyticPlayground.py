# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:55:25 2020

@author: Till
"""


import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from mayavi import mlab
import scipy.optimize as sciop
import time


"""
### meta input
P1 = np.array([0,0,-1])
P2 = np.array([0,0, 1])


TD = 0.5




### processing
# distance vector
nP = P2 - P1

# major axis intersection p1
Dist = la.norm(nP)
nvec = nP / Dist
p1   = (P1+P2)/2   +   TD/2 * nvec


# major axis
R  = (P2 - P1)
Rd = la.norm(R)

# focus
b = (P1 + P2)/2
#b = P1

# shortcuts
lm = (Rd - TD)**2
lp = (Rd + TD)**2
phim = lm / (2*Rd)
phip = lp / (2*Rd)


M = np.zeros([3, 3])
M[0, :] = np.array([ (TD/2)**2, 0, -1 ])
M[1, :] = np.array([ (Rd/2 - phim)**2, phim**2 - lm, -1 ])
M[2, :] = np.array([ (Rd/2 - phip)**2, phip**2 - lp, -1 ])
#M[1, :] = np.array([ (Rd - phim)**2, phim**2 - lm, -1 ])
#M[2, :] = np.array([ (phip)**2, phip**2 - lp, -1 ])

N = M[[0,1], 0:2]
tar = -M[[0,1], 2]
ABB = np.zeros(3)
try:
    ABB[0:2] = la.solve(N, tar)
    ABB[2] = ABB[1]
    ABB[1:] = -ABB[1:]
    K = [0]
except la.LinAlgError:
    ABB = np.array([1,0,0])
    K = [0]

D = np.diag(ABB)


"""


"""
# focus
b = P2   +   (TD*2) * nvec

# Eigenvector scaling (for hyperbolics with positive contours values)
### !!! this uses the old p1 adapted for TD between -+||R||/2
scale = lambda x: x/(Dist*2) * (Dist/(Dist - 2*x))
D = np.eye(3) * ( -np.array([0, 1, 1]).T * (1/np.dot(p1-b,p1-b))*4/3*scale(-TD)     +  np.array([1/np.dot(p1-b,p1-b), 0, 0]).T )

K = [1]

TD between -+||R||/2




## construct orthogonal eigenvector basis (eigvec 2 and 3 are orthonormal)
v1 = R/la.norm(R)
V = np.zeros([3,3])
V[:,0] = v1
V[:,1:] = sla.null_space(V.T)




r1   = Dist/2 + TD/2
r2   = Dist - r1
d2   = P1   +   2*r1**2 / Dist * nvec
rd2  = 2*r1 * ( 1 - (r1/Dist)**2 )**0.5
p2   = lambda theta: d2 + rd2*np.cos(theta) * V[:,1] + rd2*np.sin(theta) * V[:,2]
x2,y2,z2 = zip(*[p2(x) for x in np.arange(0,2*np.pi+0.1,0.1)])

delta = np.pi - (2*np.arcsin( (Dist+r2-r1)/(2*Dist) ))
d3   = P1   -   np.cos(delta) * Dist * nvec
rd3  = np.sin(delta) * Dist
p3   = lambda theta: d3 + rd3*np.cos(theta) * V[:,1] + rd3*np.sin(theta) * V[:,2]
x3,y3,z3 = zip(*[p3(x) for x in np.arange(0,2*np.pi+0.1,0.1)])

### input

# focus
b = np.array([0,0,2])

# major axis
R  = np.array([0,0,2]) 

# contours
K = list( (np.array([0, 1, 2, 3, 4, 5, 6]))**2 )
"""





"""
idx = 0
for (i1, i2) in mp:
    Rs[0,:] = 1
"""  



## Quadratic Equation
#Q = lambda x, P, D, b, sig: ( ( ( (x - b) @ P @ D @ P.T @ (x.T - b.T) ) - 1) * (1 if ((x - b) @ P[:,0] * sig < 0) else -1) )
#Q = lambda x, P, D, b: x @ D @ x.T + V[:,0] @ x
"""
print(Q(np.array([p1[0], p1[1], p1[2]]), V, D, b, np.sign(TD)))
print(Q(np.array([x2[0], y2[0], z2[0]]), V, D, b, np.sign(TD)))
print(Q(np.array([x3[0], y3[0], z3[0]]), V, D, b, np.sign(TD)))
"""



N  = np.array([[0,0,1], [0,0,-1], [0,1,0], [1,0,0], [0,-1,0]])
x_GT = np.array([0,0,-0.3]) + 0.1*np.random.random(3) * np.array([1,1,1])
Rs = la.norm(x_GT - N, axis=1)+ 0.001*np.random.random(5)

N  = np.array([[0,0,1], [0,0,-1], [0,1,0]])
#N  = np.array([[0,0,1], [0,0,-1], [0,1,0], [1,0,0], [0,-1,0]])
n  = len(N)
x_GT = np.array([0,0,-0.3])# + 0.1*np.random.random(3) * np.array([1,1,1])
Rs = la.norm(x_GT - N, axis=1)#+ 0.001*np.random.random(n)
mp = np.array([[i,j] for i in range(n) for j in range(i+1,n)])

def MetaQ(x0, N, Rs):
    
    ### pre-process data ###
    # problem size
    n   = len(N)
    dim = int(n*(n-1)/2)
    
    # get mapping array
    mp = np.array([[i,j] for i in range(n) for j in range(i+1,n)])
    
    # range differences RD (equivalent to TDOA)
    RD = Rs[mp[:,1]] - Rs[mp[:,0]]
    
    # ranges between stations 
    R  = (N[mp[:,1]] - N[mp[:,0]]).T
    Rd = la.norm(R, axis=0)
    
    
    
    ### calculate hyperbola function ###
    
    # foci of the hyperbolas (right in between stations)
    b  = ( N[mp[:,0]] + N[mp[:,1]] ) / 2
    
    # construct orthonormal eigenvector basis of hyperbolic
    v1 = R / Rd
    V = np.zeros([dim, 3, 3])
    V[:,:,0]  = v1.T
    V[:,:,1:] = np.array([sla.null_space(V[i, :, :].T) for i in range(dim)])
    
    # shortcuts for matrix
    lm = (Rd - RD)**2
    phim = lm / (2*Rd)
    #lp = (Rd + RD)**2
    #phip = lp / (2*Rd)
    
    # matrix for coefficient computation
    N = np.zeros([dim, 2, 2])
    N[:,:,:] = np.array([np.array([[ (RD[i]/2)**2, 0],\
                                   [ (Rd[i]/2 - phim[i])**2, -phim[i]**2 + lm[i]]\
                                   #[ (Rd[i]/2 - phip[i])**2, -phip[i]**2 + lp[i]]
                                       ])\
                         for i in range(dim)] )
    
    # compute coefficients by solving linear system; assuming target contour 
    # will be 1
    ABB = np.zeros([dim,3])
    ABB[:, :2] = la.solve(N, np.ones([dim, 2]))
    ABB[:, 2] = ABB[:, 1]

    # diagonal matrix for correct scaling of the eigenvectors
    D        = np.zeros([dim, 3, 3])
    D[:,:,:] = np.array( [np.diag(ABB[i,:]) for i in range(dim) ])
    
    
    
    ### set up solution
    
    # A matrix of the quadratic form
    A   = V @ D @ V.swapaxes(1,2)
    
    # lil helper to "label" data on the "wrong side" of the hyperbolic
    sig = np.sign(RD)
    
    # quadratic form F = x^T A x  - 1; but swapping sign to capture the "correct"
    # side of the hyperbolic
    def FJ(x, mode=0):
        sign_cond = np.array( [(np.dot((x - b[i]), V[i,:,0]) * sig[i] < 0) for i in range(dim)] )
        swapsign, add = zip(*[((1,1) if cond else (-1,0)) for cond in sign_cond])
        
        if mode == 0:
            R = np.array( [ \
                   ( ( np.dot((x - b[i]), np.dot(A[i,:,:], (x.T - b[i].T) ) ) ) \
                    * 1) * swapsign[i] + 0*add[i] - 1\
                    for i in range(dim)] )
        elif mode == 1:
            R = np.array( [ \
                   ( np.dot(A[i,:,:], (x - b[i]) ) ) \
                    * 1 + 0*swapsign[i]\
                for i in range(dim)] )
        return R
    
    """
    F   = lambda x: np.array( [\
                    ( ( np.dot((x - b[i]), np.dot(A[i,:,:], (x.T - b[i].T) ) ) ) \
                       * (1 if (np.dot((x - b[i]), V[i,:,0]) * sig[i] >= 0) else -1)\
                       ) - 0 \
                    for i in range(dim)] )
    # apparently DelF = 2 A x; same sign-trick applies
    J    = lambda x: np.array( [\
                    2  * ( np.dot(A[i,:,:], (x - b[i])) ) \
                       * (1 if (np.dot((x - b[i]), V[i,:,0]) * sig[i] >= 0) else -1) \
                    for i in range(dim)] )
    """
    
    
        
    ### finally, solve...
    
    xsol = sciop.least_squares(lambda x: FJ(x, 0), x0, method='dogbox')#, jac = J)#, method='lm', xtol=1e-8, x_scale='jac')
    
    # record results
    xn = xsol.x
    cost = xsol.cost
    opti = xsol.optimality
    fval = xsol.fun
    
    
    return xn, fval, FJ



n   = len(N)
dim = int(n*(n-1)/2)

timestart = time.time()
n_trials = 100
xn = np.zeros([n_trials, 3])
for idx in range(n_trials):
    xn[idx,:], fvalfinal, FJ = MetaQ(np.array([0,-idx*0.01,-idx*0.01]), N, Rs)
timeend = time.time()


print("%fus"% ((timeend - timestart) * 1000000/n_trials) )

#print("\n")
print("%.1f%%" % (sum(la.norm(xn - x_GT, axis=1) < 5e-2)/n_trials*100))



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
            F[:, i,j,k] = FJ(xvec, 0)

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
            J[:, :, i,j,k] = FJ(xvec, 1)

## plottare
mlab.close(all=True)

ck = [(0,0,1), (0,1,0), (1,0,0), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]
for idx in range(3):#range(dim):
    mlab.contour3d(x, y, z, F[idx, :,:,:], transparent=True, contours = [0], opacity=0.6, color = ck[idx])
    #mlab.quiver3d(xj, yj, zj, J[idx, 0], J[idx, 1], J[idx, 2])
    
for idx in range(n):
    mlab.points3d(N[idx, 0], N[idx, 1], N[idx, 2], color = (1,1,1), scale_factor=1e-1)

mlab.points3d(x_GT[0], x_GT[1], x_GT[2], color = (0,1,0), scale_factor=1e-1)
mlab.points3d(xn[-1, 0], xn[-1, 1], xn[-1, 2], color = (1,0,0), scale_factor=1e-1)


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
