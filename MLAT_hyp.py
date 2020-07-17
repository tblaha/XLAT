# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:40:56 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la
from constants import *
import scipy.optimize as sciop
import scipy.linalg as sla

class FeasibilityError(Exception):
    """Raised when calculation not feasibe"""
    # 1. not enough stations
    # 2. best measurements still unphysical
     
    pass


class ConvergenceError(Exception):
    """Raised when iterative procedure failes to converge"""
    
    pass


def fun(N, mp, x):
    """
    Returns the range differences based on location and station locations. At 
    the solution x_sol, the vector fval

    Parameters
    ----------
    N : array(n, 3)
        Station location matrix (cartesian).
    mp : array(n*(n-1)/2, 2)
        Mapping matrix, maps a time difference index to the indices of the 
        subtrahend and minuend. Rows correspond to the Axis 0 of fval and the
        TDOA measurements. The first column is the subtrahend, second column
        is the minuend.
    x : array(1, 3)
        Location.

    Returns
    -------
    fval : array(n*(n-1)/2, 1)
        Differences in Ranges (distances station-aircraft) between each of the
        stations in meter
    """
    
    # number of TDOA measurements
    n = np.size(mp, axis=0)
    
    # prealloc solution vector
    fval     = np.zeros([n+1, 1])
    fval[-1] = la.norm(x)  # altitude
    
    # iterate over the items
    for i in np.arange(n):
        # Minuend (second column)  - Subtractor (first column)
        # the mp map contains ids, but they start at 1, hence the -1
        fval[i] = + la.norm( N[(mp[i,1]-1), :] - x )\
                  - la.norm( N[(mp[i,0]-1), :] - x )
    
        
    
    return fval


def rho_alt(e, beta):
    """
    loss function to "prefer" the planes baroAlt to the (possibly conflicting)
    hyperbolic surfaces. This is done by scaling up the error of the last
    element of e == f**2 == (fun(N, mp, x) - T*C0)**2, which corresponds to 
    the altitude error
    https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.optimize.least_squares.html

    Parameters
    ----------
    e : array(3, m)
        Error components of the objective functions.
    beta : scalar > 0
        Scaling factor to promote convergence of the altitude error. If large 
        than 1, the altitude error will "mean more" to the NLLSq algorithm

    Returns
    -------
    A : array(3, m)
        as described in docs it contains
        array[0, :]: rho(e)
        array[1, :]: rho'(e) = diag(Del * rho(e)) (elementwise derivatives)
        array[2, :]: rho''(e) = diag(Del**2 * rho(e)) 
            (elementwise seconds derivatives)
    """
    A     = np.empty([3, e.size])
    scale = np.concatenate( (np.ones([e.size-1]), [beta]) )
    A[0]  = e * scale 
    A[1]  = scale # derivatives wrt elements of e
    A[2]  = np.zeros_like(e) # seconds derivatives
    return A
    
    


def delta(n, x0):
    """
    local gradient vector to the range around station n around x0 in cartesian

    Parameters
    ----------
    n : array(1,3)
        station location (cartesian)
    x0 : array(1,3)
        linearization point

    Returns
    -------
    d : array(1,3)
        local gradient vector

    """

    # derivative of sqrt((x-xn)**2 + (y-yn)**2 + (z-zn)**2) around (x0, y0, z0)
    
    D  = np.array([(x0[0] - n[0]), (x0[1] - n[1]), (x0[2] - n[2])])
    nD = la.norm(D)
    
    d = np.array([ D[0], D[1], D[2] ]) / nD
    
    return d



def Jac(N, mp, x0):
    """
    Returns the Jacobian matrix of the linearization around x0 of fun(...). 
    Useful for the first order approximation:
        
    fun(N, mp, x) = J(N, mp, x0) * (x - x0) + fun(N, mp, x0)
    
    J(N, mp, x0) = [ [Del fun(n2-n1, mp, x0)], \
                     [Del fun(n3-n1, mp, x0)], ... ]
    J(N, mp, x0) = [ [delta(n2, x0) - delta(n1, x0)], \
                     [delta(n3, x0) - delta(n1, x0)], ... ]    
    
    Parameters
    ----------
    N : array(n, 3)
        Station location matrix (cartesian).
    mp : array(n*(n-1)/2, 2)
        Mapping matrix, maps a time difference index to the indices of the 
        subtrahend and minuend. Rows correspond to the Axis 0 of fval and the
        TDOA measurements. The first column is the subtrahend, second column
        is the minuend.
    x0 : array(1, 3)
        Location around which to linearize.

    Returns
    -------
    J : matrix(n*(n-1)/2, 3)
        Jacobian matrix.

    """

    # number of TDOA measurements
    n = np.size(mp, axis=0)

    # prealloc matrix
    J = np.matrix(np.zeros([n + 1, 3]))
    J[-1] = np.ones([1, 3]) * 1/la.norm(x0)

    # iterate over the rows
    for i in np.arange(n):
        # Minuend (second column)  - Subtractor (first column)
        # the mp map contains ids, but they start at 1, hence the -1
        J[i, :] = (delta(N[(mp[i, 1] - 1), :], x0)
                   - delta(N[(mp[i, 0] - 1), :], x0))

    return J



def iterx(N, T, xn):
    """
    Old fashioned NLLS iteration scheme; not currently used

    Parameters
    ----------
    N : array(n, 3)
        Station location matrix (cartesian).
    T : array(n*(n-1)/2, 1)
        TDOA vector.
    xn : array(1, 3)
        current location to linearize around.

    Returns
    -------
    xnplus1 : array(1, 3)
        next location according to the iteration scheme.

    """
    
    global C0
    
    # get Jacobian for linearization around previous value xn
    J = Jac(N, xn)
    
    # invert equation above for (xnplus1 - xn)
    delx = la.pinv(J) @ (T*C0 - fun(N, xn))
    
    # next value by adding the current linearization point to the solution to
    # the linearization problem
    xnplus1 = np.array(xn + delx.T) 
    
    return xnplus1[0]
    



def getHyperbolic(N, mp, dim, RD, R, Rd):
    
    # foci of the hyperbolas (right in between stations)
    b  = (N[mp[:, 0]] + N[mp[:, 1]]) / 2
    
    # construct orthonormal eigenvector basis of hyperbolic
    v1 = R.T / Rd
    V = np.zeros([dim, 3, 3])
    V[:, :, 0]  = v1.T
    V[:, :, 1:] = np.array([sla.null_space(V[i, :, :].T) 
                            for i in range(dim)
                            ])
    
    # shortcuts for matrix
    lm = (Rd - RD)**2
    phim = lm / (2*Rd)
    #lp = (Rd + RD)**2
    #phip = lp / (2*Rd)
    
    # matrix for coefficient computation
    N = np.zeros([dim, 2, 2])
    N[:, :, :] = np.array([
        np.array([[ (RD[i]/2)**2, 0],
                  [ (Rd[i]/2 - phim[i])**2, -phim[i]**2 + lm[i]]
                  #[ (Rd[i]/2 - phip[i])**2, -phip[i]**2 + lp[i]]
                  ])
        for i in range(dim)
        ])
    
    # compute coefficients by solving linear system; assuming target contour 
    # will be 1
    ABB = np.zeros([dim, 3])
    ABB[:, :2] = la.solve(N, np.ones([dim, 2]))
    ABB[:, 2] = ABB[:, 1]

    # diagonal matrix for correct scaling of the eigenvectors
    D        = np.zeros([dim, 3, 3])
    D[:,:,:] = np.array([np.diag(ABB[i, :])
                         for i in range(dim)
                         ])
    
    # A-matrix of the quadratic form
    A   = V @ D @ V.swapaxes(1,2)
    
    return A, V, D, b




def getSphere(rho_baro):
    
    # trivial eigenvectors
    V = np.eye(3)
    
    # equal eigenvalues
    D = np.diag(1 / (rho_baro ** 2) * np.ones(3))
    
    # trivial center point
    b = np.zeros(3)
    
    # A matrix
    #A = V @ D @ V.T
    A = D # because V is eye-dentity anyway
    
    return A, V, D, b
    


def FJ(x, A, b, dim, V, sig, mode = 0):
    
    # discriminant for the sign flip
    sign_cond = np.array( [(np.dot((x - b[i]), V[i,:,0]) * sig[i] < 0) 
                           for i in range(dim)] )
    swapsign, flip = zip(*[((1,0) if cond else (-1,1)) for cond in sign_cond])
        
    if mode == 0:
        Ret = np.array( [ \
               ( ( np.dot((x - b[i]), np.dot(A[i,:,:], (x.T - b[i].T) ) ) ) 
                * 1) * swapsign[i] - 1
                for i in range(dim)] )
    elif mode == 1:
        delta = np.array( [ \
                    np.dot(A[i,:,:], (x - b[i]) ) \
                    for i in range(dim)] )
        Ret = delta - np.array([ \
                flip[i] * 2 * np.dot(delta[i], V[i,:,0]) * V[i,:,0] \
                for i in range(dim)])
            
    return Ret

        
def SelectMeasurements(RD_sc, Kmin):
    # determine usable measurements
    m_use = (abs(RD_sc) <= 0.99)
    
    # error if not enough
    if sum(m_use) < Kmin:
        raise FeasibilityError("Not enough phyiscal\
                               (lambda < 0.99) measurements")
        
    return m_use
        
        
def MetaQ(N, mp, Rs, rho_baro=-1):
    
    ### pre-process data ###
    # range differences RD (equivalent to TDOA)
    RD   = (Rs[mp[:,1]] - Rs[mp[:,0]]) # meters
    
    # vectors and ranges between stations 
    R  = (N[mp[:,1]] - N[mp[:,0]])
    Rn = la.norm(R, axis=1)
    
    # do we have a baro altitude to use?
    use_baro = (rho_baro > 0)
    
    
    ### check quality of measurements and discard accordingly
    # scale measurement to distance between stations
    RD_sc = RD/Rn
    m_use = SelectMeasurements(RD_sc, 3 - int(use_baro))
    
    # update all vectors
    mp    = mp[m_use]
    RD    = RD[m_use]
    R     = R [m_use]
    Rn    = Rn[m_use]
    RD_sc = RD_sc[m_use]
    
    # little helper for discriminating between solutions later on
    RDsi = np.sign(RD) # don't think it's necessary anymore
    
    # determine problem size
    dim = len(mp)
    
    
    
    ### calculate quadratic form
    if rho_baro < 0:
        # don't rely on altitude info
        A = np.zeros([dim, 3, 3])
        V = np.zeros([dim, 3, 3])
        D = np.zeros([dim, 3, 3])
        b = np.zeros([dim, 3])
        A[:,:,:], V[:,:,:], D[:,:,:], b[:,:] =\
            getHyperbolic(N, mp, dim, RD, R, Rn)
    else:
        # use aultitude info
        A = np.zeros([dim+1, 3, 3])
        V = np.zeros([dim+1, 3, 3])
        D = np.zeros([dim+1, 3, 3])
        b = np.zeros([dim+1, 3])
        A[:-1,:,:], V[:-1,:,:], D[:-1,:,:], b[:-1,:] =\
            getHyperbolic(N, mp, dim, RD, R, Rn)
        
        A[-1,:,:],  V[-1,:,:],  D[-1,:,:],  b[-1,:]  =\
            getSphere(rho_baro)
    
    
    
    ### generate x0
    x0_idx = np.where(abs(RD_sc) == np.min(abs(RD_sc)))[0][0]
    x0 = N[mp[x0_idx,0],:] + \
        (0.5+RD_sc[x0_idx]/2) * (N[mp[x0_idx,1],:] - N[mp[x0_idx,0],:])
    #x0 = np.array([0,-0.5,-0.5])
    
    ### solve and record
    xsol = sciop.least_squares(lambda x: FJ(x, A, b, dim, V, RDsi, mode=0),\
                               x0, method='lm', x_scale='jac')
    #xsol = sciop.least_squares(lambda x: FJ(x, A, b, dim, V, RDsi, mode=0),\
    #                     x0, method='lm', x_scale='jac',\
    #                     jac = lambda x: FJ(x, A, b, dim, V, RDsi, mode = 1))
    
    inDict = {'A':A, 'b':b, 'V':V, 'D':D, 'dim':dim, 'RDsi':RDsi}
        
    #EvaluateResult(xsol, dim)
    xn   = xsol.x
    cost = xsol.cost
    fval = xsol.fun
    opti = xsol.optimality
    succ = xsol.success
    
    return xn, cost, fval, opti, succ, inDict



def NLLS_MLAT(MR, NR, idx, solmode = 1):
    """
    Wrapper of the iterative non-linear least squares calculation for ac 
    position including preprocessing of the Station coordinates, TDOA and 
    initial guess generation

    Parameters
    ----------
    MR : pd.DataFrame
        Measurements.
    NR : pd.DataFrame
        Stations.
    idx : scalar
        MR.id datapoint to solve.
    solmode : scalar
        the solution procedure to use:
            0 --> self coded NLLSq
            1 --> scipy.optimize.least_squares

    Returns
    -------
    xn : array(1, 3)
        Solution in the least square sense in CARTESIAN.
    xn_sp : array(1, 3)
        Solution in the least square sense in LAT, LONG, h(feet).
    fval : scalar
        Function value minus TDOA ranges at the found solution.

    """
    global X,Y,Z, SP2CART, CART2SP, C0
    
    
    
    ### preprocess stations and measurements
    
    
    # find stations
    stations = np.concatenate(MR[MR.id == idx].n.to_numpy())
    n = MR[MR.id == idx].M.iloc[0]
    
    
    # get station locations and convert to cartesian
    lats  = NR.set_index('n').loc[stations].\
                reset_index(inplace=False).lat.to_numpy()
    longs = NR.set_index('n').loc[stations].\
                reset_index(inplace=False).long.to_numpy()
    geoh  = NR.set_index('n').loc[stations].\
        reset_index(inplace=False).geoAlt.to_numpy()
    
    N = np.array([X(lats, longs, geoh),
                  Y(lats, longs, geoh),
                  Z(lats, longs, geoh)
                  ]).T


    
    
    ### convert unix time stamps of stations to TDOA*C0 differential ranges
    
    # grab station TOA; get nanoseconds into seconds
    secs = np.concatenate(MR[MR.id == idx].ns.to_numpy())*1e-9 

    # pre alloc 
    b         = np.zeros([dim+1, 1])
    mut_dists = np.zeros([dim, 1])
    mp        = np.zeros([dim, 2]) # Mapping matrix, maps a time difference 
                                   # index to the indices of the subtrahend \
                                    # and minuend.

    # iterate over the possible differences between 2 stations out of n
    index = 0
    for i in np.arange(1,n):
        for j in np.arange(i+1,n+1):
            b[index, :]      = secs[j-1] - secs[i-1]
            mut_dists[index] = la.norm(N[j-1]-N[i-1])
            mp[index, :]     = np.array([i, j])
            index            = index + 1
    
    # make mapping contain on ints (for indexing with it later)
    mp = np.vectorize(int)(mp)
    
    # add altitude (radius) target from baroAlt (light speed normalized)
    b[-1]       = (MR[MR.id == idx].baroAlt + R0) / C0
    
    
    ### assess quality of measurements
    
    # scaled TDOA ranges by distances between stations
    mut_dists_sc = b[0:-1]*C0/mut_dists
    
    # select the K ones with the lowest ("sort") number --> most planar 
    # surfaces as opposed to hyperboloids. This is a bool-array
    K = 2
    active_pnts = np.in1d(np.arange(dim), \
                          np.argsort(abs(mut_dists_sc).T[0])[:K])
    
    # abs(mut_dists_sc) > 1 are unphysical (and most likely unsolvable, too)
    if np.max(abs(mut_dists_sc[active_pnts])) > 1:
        raise FeasibilityError("Best measurements contain \
                               unphyiscal (lambda > 1) points")
    
    
    ### actual solution process
    
    # generate initial guess --> find most suitable 2 stations to put the
    # initial guess in between. This satisfies at least 1 differential range
    # equation and should help convergences. Suitable means smallest TDOA small
    # comparted to distance between the stations (means hyperbolic surfaces
    # become more planar)
    x0_idx = np.where(abs(mut_dists_sc) == np.min(abs(mut_dists_sc)))[0][0]
    x0 = N[mp[x0_idx,0]-1,:] + \
              (0.5+mut_dists_sc[x0_idx]/2) * \
                  (N[mp[x0_idx,1]-1,:] - N[mp[x0_idx,0]-1,:])
    """ previous attempts
    #x0 = np.array([0,0,0])
    #x0 = SP2CART(MR[MR.id == idx].lat.iloc[0],\
                  MR[MR.id == idx].long.iloc[0],\
                  MR[MR.id == idx].geoAlt.iloc[0])
    #x0 = np.mean(N, axis=0)
    #x0 = np.mean(N, axis=0) + SP2CART(lats[0], longs[0], -R0 + 1e4)
    #x0 = np.mean( [ N[0,:], N[1,:], N[2,:] ], axis=0 )
    """
    
    if solmode == 0:
        # Old fashioned NLLS
        itermax = 20 # max iterations
    
        it = 0
        xnminus1 = np.array([1,1,1])*1e10
        xn = x0
        while it < itermax and la.norm(xn-xnminus1) > 1:
            xnminus1 = xn
            xn = iterx(N,b,xnminus1)
            
            it = it + 1
        
        if la.norm(xn-xnminus1) > 1:
            raise ConvergenceError("Error norm not below threshold")
            
    elif solmode == 1:
        llsq_active = np.concatenate( (active_pnts, [True]) )
        # use scipy's LSQ solver based on LM with custom Jacobian
        # only solve at active_points
        sol = sciop.least_squares(\
            lambda x: np.array( \
            fun(N, mp, x).T)[0][llsq_active] - b.T[0][llsq_active] * C0, \
            x0, \
            jac=lambda x: np.array(Jac(N, mp, x)[llsq_active]), \
            #method='dogbox', x_scale='jac', loss='linear',tr_solver='exact',\
            #method='dogbox', x_scale='jac', loss='soft_l1', f_scale=1e4, \
                #tr_solver='exact', \
            #method='dogbox', x_scale='jac', loss=lambda x: rho_alt(x, 1e0),\
                #f_scale=1e4, tr_solver='exact', \
            method='lm', x_scale='jac', \
            max_nfev=40, xtol=1e-8, gtol=1e-8, ftol=None, \
            verbose=0)
        xn = sol.x
        
        # yep, python is ridiculous... 
        #this selects to used stations and adds it to the dataframe
        try:
            MR.n_used.loc[idx-1] = \
                tuple(\
            np.array(MR.loc[MR.id == idx, 'n'].\
                     to_numpy()[0])[np.unique(mp[active_pnts])-1])
        except IndexError:
            print("asdf")
            
        MR.loc[MR.id == idx, "cost"] = sol.cost
        MR.loc[MR.id == idx, "nfev"] = sol.nfev
        
        if not sol.success or sol.cost > 2e6:
            raise ConvergenceError("Not happy with sciop.least_squares' deed")
    

    return xn, CART2SP(xn[0], xn[1], xn[2]), (fun(N, mp, xn)-b*C0)[llsq_active]
    

    