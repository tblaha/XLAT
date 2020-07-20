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


class MLATError(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return repr(self.code)




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


def FJ(x, A, b, dim, V, RD, mode=0):
    sig = np.sign(RD)

    # discriminant for the sign flip
    sign_cond = np.array([(np.dot((x - b[i]), V[i, :, 0]) * sig[i] < 0)
                          for i in range(dim)
                          ])
    swapsign, flip = zip(*[((1, 0) if cond else (-1, 1))
                           for cond in sign_cond
                           ])

    if mode == 0:
        # compute function including the sign swapping
        # Ret = np.array([
        #     (((np.dot((x - b[i]),
        #               np.dot(A[i, :, :], (x.T - b[i].T)))
        #        + 0*swapsign[i] - 1)**2)**(0.5)
        #      - flip[i] * np.dot((x - b[i]), V[i, :, 0])*25
        #      ) for i in range(dim)
        #     ])
        Ret = np.array([
            ((np.dot((x - b[i]),
                      np.dot(A[i, :, :], (x.T - b[i].T)))
               +0* swapsign[i] - 1)
             ) for i in range(dim)
            ])
    elif mode == 1:
        # compute Jacobian with the v1 compunent mirroring
        # delta = np.array([
        #                   np.dot(A[i, :, :], (x - b[i]))
        #                   for i in range(dim)
        #                   ])
        # Ret = delta - 2 * np.array(
        #     [flip[i] * np.dot(delta[i], V[i, :, 0]) * V[i, :, 0]
        #      for i in range(dim)
        #      ])
        Ret = np.array([2 * np.dot(A[i, :, :], (x - b[i])) 
                        for i in range(dim)
                        ])
    return Ret


def FJsq(x, A, b, dim, V, RD, mode=0):
    f = np.array([  # (x-b)A(x-b)^T - 1)
        (np.dot((x - b[i]), np.dot(A[i, :, :], (x.T - b[i].T))) - 1)
        for i in range(dim)
        ])
    q = np.array([  # 2 * (x-b)A
        2 * np.dot(x - b[i], A[i, :, :])
        for i in range(dim)
        ])
    p = 8 * np.sign(RD) / (np.abs(RD) + 1e-12)  # looks stupid, is robust
    d = np.array([
        np.dot((x - b[i]),  V[i, :, 0])
        for i in range(dim)
        ])
    
    cond = (f > 0) & (d * p > 0)
    flip = [-1 if c else 1 for c in cond]
    mcc = lambda x: np.heaviside(x, 0) * x
    
    ftilde = f * flip - mcc(d * p)
    qtilde = (q - mcc(p * V[:, :, 0].T).T)
    qtilde = q
    
    if mode == -1:
        Ret = ftilde
    elif mode == 0:
        Ret = 0.5*np.sum(ftilde**2)
    elif mode == 1:
        Ret = np.sum((ftilde * qtilde.T).T,
                     axis=0)
    elif mode == 2:
        Ret = 4 * np.sum(np.array([
            0.5 * ftilde[i] * A[i, :, :] 
            + np.matrix(qtilde[i]).T @ np.matrix(qtilde[i])
            for i in range(dim)
            ]), axis=0)
    
    return Ret


""" vanilla quadratic form including jacobian and hessian
def FJsq(x, A, b, dim, V, RD, mode=0):
    sig = np.sign(RD)
    if mode == 0:
        Ret = np.sum(0.5*np.array([
             (np.dot((x - b[i]), np.dot(A[i, :, :], (x.T - b[i].T))) - 1)**2
              for i in range(dim)
             ]))
    elif mode == 1:
        Ret = np.sum(np.array([
            (np.dot((x - b[i]), np.dot(A[i, :, :], (x.T - b[i].T))) - 1)
            * (2 * np.dot(A[i, :, :], (x - b[i])))
            for i in range(dim)
            ]), axis=0)
    elif mode == 2:
        Ret = np.sum(np.array([
             (np.dot((x - b[i]), np.dot(A[i, :, :], (x.T - b[i].T))) - 1)
             * 2 * A[i, :, :]
             + 4 * np.cross(np.dot(A[i, :, :], x - b[i]),
                            np.dot(x - b[i], A[i, :, :])
                            )
              for i in range(dim)
             ]), axis=0)
    return Ret
"""


def CON(x, C, mode=0):
    # value. jacobian or hessian of the constraint
    if mode == 0:
        sol = x @ C @ x
    elif mode == 1:
        sol = 2 * (C @ x)
    elif mode == 2:
        sol = 2 * C

    return sol


def GenMeasurements(N, n, Rs):

    # ### pre-process data ###
    # mapping of the stations to the differences
    mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])

    # range differences RD (equivalent to TDOA)
    RD = (Rs[mp[:, 1]] - Rs[mp[:, 0]])  # meters

    # vectors and ranges between stations
    R = (N[mp[:, 1]] - N[mp[:, 0]])
    Rn = la.norm(R, axis=1)

    # Scaled measurements
    RD_sc = RD/Rn

    # determine usable measurements
    m_use = ((abs(RD_sc) <= 0.99)
             & (abs(Rn) > 2000)
             )

    # error if not enough stations left
    Kmin = 3
    if len(RD_sc) < Kmin:
        raise MLATError(1)
        # raise MLATError("Not enough measurements available")
    elif sum(m_use) < Kmin:
        # raise FeasibilityError("Not enough phyiscal (lambda < 0.99) or \
        #                        usable (R_stations < 1km) measurements \
        #                        available")
        raise MLATError(2)

    # update all vectors
    mp = mp[m_use]
    RD = RD[m_use]
    R = R[m_use]
    Rn = Rn[m_use]
    RD_sc = RD_sc[m_use]

    return mp, RD, R, Rn, RD_sc


def genx0(N, mp, RD_sc, rho_baro):
    # find index of most "central" measurement (least difference of Range
    # difference relative to station distance)
    x0_idx = np.where(abs(RD_sc) == np.min(abs(RD_sc)))[0][0]

    # find the x0 as the position on the line between those 2 stations that
    # also satisfies the hyperbola
    x0 = N[mp[x0_idx, 0], :] + \
        (0.5 + RD_sc[x0_idx] / 2) * (N[mp[x0_idx, 1], :] - N[mp[x0_idx, 0], :])

    if rho_baro > 0:
        std_mean = np.mean(N, axis=0)
        x0 = std_mean / la.norm(std_mean) * rho_baro
    
    return x0


def CheckResult(sol, dim):
    # solver didn't converge
    if not sol.success:
        raise MLATError(30 + sol.status)
    # if sol.optimality > 1:
    #     raise MLATError(4)
    # if sol.cost > 1:
    #     raise MLATError(5)

    xn   = sol.x
    # opti = sol.optimality
    opti = 0
    cost = sol.fun
    nfev = sol.nfev
    # niter = sol.niter
    niter = sol.nit

    return xn, opti, cost, nfev, niter


def MLAT(N, n, Rs, rho_baro=-1):

    # ### check quality of measurements and discard accordingly
    # scale measurement to distance between stations
    mp, RD, R, Rn, RD_sc = GenMeasurements(N, n, Rs)

    # determine problem size
    dim = len(mp)

    # ### calculate quadratic form
    A, V, D, b = getHyperbolic(N, mp, dim, RD, R, Rn)
    if rho_baro >= 0:
        # use aultitude info
        C, __, __, __ = getSphere(rho_baro)

        # define equality constraint
        cons = sciop.NonlinearConstraint(lambda x: R0*CON(x, C, mode=0), R0, R0,
                                         jac=lambda x: R0*CON(x, C, mode=1),
                                         hess=lambda x, __: CON(x, C, mode=2),
                                         )
    else:
        cons = ()

    # ### setup problem
    # objective functions
    def fun(x):
        return FJ(x, A, b, dim, V, RD, mode=0)

    def funlsq(x):
        return np.sum(fun(x)**2)

    # ### generate x0
    x0 = genx0(N, mp, RD_sc, rho_baro)

    # ### solve and record
    """xsol = sciop.least_squares(lambda x:
    #                 FJ(x[:-1], A, b, dim, V, RDsi, mode=0)
    #                 - x[-1] * CON(x[:-1], C),
    #                 x0, method='trf', x_scale='jac',
    #                 verbose=0, max_nfev=50)
    """
    xlist = []
    sol = sciop.minimize(
                    lambda x: FJsq(x, A, b, dim, V, RD, mode=0),
                    x0,
                    #jac=lambda x: FJsq(x, A, b, dim, V, RD, mode=1),
                    #hess=lambda x: FJsq(x, A, b, dim, V, RD, mode=2),
                    method='SLSQP',
                    #tol=1e-9,
                    constraints=cons,
                    options={'maxiter': 50,
                             },
                    callback=lambda xk: xlist.append(xk)
                    )
    
    # check result for consistency and return the final solution or nan
    xn, opti, cost, nfev, niter = CheckResult(sol, dim)

    # build diagnostic struct
    inDict = {'A': A, 'b': b, 'V': V, 'D': D, 'dim': dim, 'RD': RD, 'xn': xn, 
              'fun': lambda x, m: FJsq(x, A, b, dim, V, RD, mode=m), 
              'xlist': xlist}

    return xn, opti, cost, nfev, niter, RD, inDict


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

    pdcross = MR.loc[idx, :]

    # ### preprocess stations and measurements
    # get number of stations
    n = pdcross['M']

    # get station locations and convert to cartesian
    stations = np.array(pdcross['n'])
    lats, longs, geoh \
        = NR.loc[stations, ['lat', 'long', 'geoAlt']].to_numpy().T
    N = SP2CART(lats, longs, geoh).T

    # ### get unix time stamps of stations
    Rs = np.array(pdcross['ns']) * 1e-9 * C0  # meters

    # baro radius
    r_baro = pdcross['baroAlt'] + R0  # meters

    # actually solve
    try:
        xn, opti, cost, nfev, niter, RD, inDict =\
            MLAT(N, n, Rs, rho_baro=r_baro)
            
        xn_sph = CART2SP(xn[0], xn[1], xn[2])
        
        MR.at[idx, "xn_sph_lat"] = xn_sph[0]
        MR.at[idx, "xn_sph_long"] = xn_sph[1]
        MR.at[idx, "xn_sph_alt"] = xn_sph[2]
        
        MR.at[idx, "fval"] = cost
        MR.at[idx, "optimality"] = opti
        MR.at[idx, "nfev"] = nfev
        MR.at[idx, "niter"] = niter
        
        MR.at[idx, "MLAT_status"] = 0
        
    except MLATError as e:
        xn_sph = np.zeros(3)
        xn_sph[:] = np.nan
        MR.at[idx, "MLAT_status"] = e.code
        inDict = {}
            
    return xn_sph, inDict
