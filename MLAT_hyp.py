# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:40:56 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la
from constants import C0, R0, X, Y, Z, SP2CART, CART2SP
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
    b = (N[mp[:, 0]] + N[mp[:, 1]]) / 2

    # construct orthonormal eigenvector basis of hyperbolic
    v1 = R.T / Rd
    V = np.zeros([dim, 3, 3])
    V[:, :, 0] = v1.T
    V[:, :, 1:] = np.array([sla.null_space(V[i, :, :].T)
                            for i in range(dim)
                            ])

    # shortcuts for matrix
    lm = (Rd - RD)**2
    phim = lm / (2*Rd)
    # lp = (Rd + RD)**2
    # phip = lp / (2*Rd)

    # matrix for coefficient computation
    N = np.zeros([dim, 2, 2])
    N[:, :, :] = np.array([
        np.array([[(RD[i]/2)**2, 0],
                  [(Rd[i]/2 - phim[i])**2, -phim[i]**2 + lm[i]]
                  # [ (Rd[i]/2 - phip[i])**2, -phip[i]**2 + lp[i]]
                  ])
        for i in range(dim)
        ])

    # compute coefficients by solving linear system; assuming target contour
    # will be 1
    ABB = np.zeros([dim, 3])
    for i, Nitem in enumerate(N):
        try:
            ABB[i, :2] = la.solve(Nitem, np.ones([2]))
            ABB[i, 2] = ABB[i, 1]
        except la.LinAlgError:
            ABB[i] = np.array([1, 0, 0])
    
    # diagonal matrix for correct scaling of the eigenvectors
    D = np.zeros([dim, 3, 3])
    D[:, :, :] = np.array([np.diag(ABB[i, :])
                           for i in range(dim)
                           ])

    # A-matrix of the quadratic form
    A = V @ D @ V.swapaxes(1, 2)

    return A, V, D, b


def getSphere(rho_baro):

    # trivial eigenvectors
    V = np.eye(3)

    # equal eigenvalues
    D = np.diag(1 / (rho_baro ** 2) * np.ones(3))

    # trivial center point
    b = np.zeros(3)

    # A matrix
    # A = V @ D @ V.T
    A = D  # because V is eye-dentity anyway

    return A, V, D, b


def FJ(x, A, b, dim, V, sig, mode=0):

    # discriminant for the sign flip
    sign_cond = np.array([(np.dot((x - b[i]), V[i, :, 0]) * sig[i] < 0)
                          for i in range(dim)
                          ])
    swapsign, flip = zip(*[((1, 0) if cond else (-1, 1))
                           for cond in sign_cond
                           ])

    if mode == 0:
        # compute function including the sign swapping
        Ret = np.array([
                        (((np.dot((x - b[i]),
                                np.dot(A[i, :, :], (x.T - b[i].T)))
                         + 0*swapsign[i] - 1)**2)**(0.5)
                         - flip[i] * np.dot((x - b[i]), V[i, :, 0])*25
                         ) for i in range(dim)
                        ])
    elif mode == 1:
        # compute Jacobian with the v1 compunent mirroring
        delta = np.array([
                          np.dot(A[i, :, :], (x - b[i]))
                          for i in range(dim)
                          ])
        Ret = delta - 2 * np.array(
            [flip[i] * np.dot(delta[i], V[i, :, 0]) * V[i, :, 0]
             for i in range(dim)
             ])
    else:
        raise RuntimeError("mode must be one of 0 or 1 for fun or jacobian")

    return Ret


def SelectMeasurements(RD_sc, Kmin):
    # determine usable measurements
    m_use = (abs(RD_sc) <= 0.99)

    # error if not enough
    if sum(m_use) < Kmin:
        raise FeasibilityError("Not enough phyiscal\
                               (lambda < 0.99) measurements available")

    return m_use


def MetaQ(N, mp, Rs, rho_baro=-1):

    # ### pre-process data ###
    # range differences RD (equivalent to TDOA)
    RD = (Rs[mp[:, 1]] - Rs[mp[:, 0]])  # meters

    # vectors and ranges between stations
    R = (N[mp[:, 1]] - N[mp[:, 0]])
    Rn = la.norm(R, axis=1)

    # do we have a baro altitude to use?
    use_baro = (rho_baro > 0)

    # ### check quality of measurements and discard accordingly
    # scale measurement to distance between stations
    RD_sc = RD/Rn
    m_use = SelectMeasurements(RD_sc, 3 - int(use_baro))

    # update all vectors
    mp = mp[m_use]
    RD = RD[m_use]
    R = R[m_use]
    Rn = Rn[m_use]
    RD_sc = RD_sc[m_use]

    # little helper for discriminating between solutions later on
    RDsi = np.sign(RD)  # don't think it's necessary anymore

    # determine problem size
    dim = len(mp)

    # ### calculate quadratic form
    if rho_baro < 0:
        # don't rely on altitude info
        A = np.zeros([dim, 3, 3])
        V = np.zeros([dim, 3, 3])
        D = np.zeros([dim, 3, 3])
        b = np.zeros([dim, 3])
        A[:, :, :], V[:, :, :], D[:, :, :], b[:, :] =\
            getHyperbolic(N, mp, dim, RD, R, Rn)
    else:
        # use aultitude info
        A = np.zeros([dim+1, 3, 3])
        V = np.zeros([dim+1, 3, 3])
        D = np.zeros([dim+1, 3, 3])
        b = np.zeros([dim+1, 3])
        A[:-1, :, :], V[:-1, :, :], D[:-1, :, :], b[:-1, :] =\
            getHyperbolic(N, mp, dim, RD, R, Rn)

        A[-1, :, :], V[-1, :, :], D[-1, :, :], b[-1, :] =\
            getSphere(rho_baro)

    # ### generate x0
    x0_idx = np.where(abs(RD_sc) == np.min(abs(RD_sc)))[0][0]
    x0 = N[mp[x0_idx, 0], :] + \
        (0.5 + RD_sc[x0_idx] / 2) * (N[mp[x0_idx, 1], :] - N[mp[x0_idx, 0], :])
    # x0 = np.array([0,-0.5,-0.5])

    # ### solve and record
    xsol = sciop.least_squares(lambda x: 
                    FJ(x, A, b, dim, V, RDsi, mode=0),
                    x0, method='trf', x_scale='jac',
                    verbose=2)
    # xsol = sciop.least_squares(lambda x: FJ(x, A, b, dim, V, RDsi, mode=0),
    #                     x0, method='lm', x_scale='jac',\
    #                     jac = lambda x: FJ(x, A, b, dim, V, RDsi, mode = 1)
    #                     )

    inDict = {'A': A, 'b': b, 'V': V, 'D': D, 'dim': dim, 'RDsi': RDsi}

    # EvaluateResult(xsol, dim)
    xn = xsol.x
    cost = xsol.cost
    fval = xsol.fun
    opti = xsol.optimality
    succ = xsol.success

    return xn, cost, fval, opti, succ, inDict


def NLLS_MLAT(MR, NR, idx, solmode=1):
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
    global X, Y, Z, SP2CART, CART2SP, C0, R0

    # ### preprocess stations and measurements
    # find stations
    stations = np.array(MR.at[idx, 'n'])
    n = MR.at[idx, 'M']

    # get station locations and convert to cartesian
    lats, longs, geoh \
        = NR.loc[stations, ['lat', 'long', 'geoAlt']].to_numpy().T
    N = SP2CART(lats, longs, geoh).T
    
    # find number of TDOA measurements available
    dim = int(n*(n-1)/2)
    if dim < 3:
        raise FeasibilityError("not enough stations")
    
    # mapping of the stations to the differences
    mp = np.zeros([dim, 2])
    mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
    
    # ### convert unix time stamps of stations to TDOA*C0 differential ranges
    # grab station TOA
    secs = np.array(MR.at[idx, 'ns']) * 1e-9  # get nanoseconds into seconds
    Rs = secs * C0  # meters
    
    # baro radius
    h_baro = MR.at[idx, 'baroAlt']
    
    # actually solve
    # xn, cost, fval, opti, succ, inDict = MetaQ(N, mp, Rs, rho_baro=R0+h_baro)
    xn, cost, fval, opti, succ, inDict = MetaQ(N, mp, Rs, rho_baro=-1)

    return xn,\
        CART2SP(xn[0], xn[1], xn[2]),\
        fval
