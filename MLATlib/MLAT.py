# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:40:56 2020

@author: Till
"""

from .helper import C0, SP2CART, CART2SP, WGS84, R1

import numpy as np
import numpy.linalg as la

import scipy.optimize as sciop
import scipy.linalg as sla
import scipy.stats as scist


class MLATError(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return repr(self.code)


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

    singularity = np.zeros(dim).astype(bool)

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
            singularity[i] = True

    # diagonal matrix for correct scaling of the eigenvectors
    D = np.zeros([dim, 3, 3])
    D[:, :, :] = np.array([np.diag(ABB[i, :])
                           for i in range(dim)
                           ])

    # A-matrix of the quadratic form
    A = V @ D @ V.swapaxes(1, 2)

    return A, V, D, b, singularity


def FJsq(x, A, b, dim, V, RD, Rn, mode=0, singularity=0):
    def bias_sign(x):
        # return np.sign(x)
        return np.heaviside(x, 1) * 2 - 1

    if not np.isscalar(singularity):
        RD[singularity] = 0
    else:
        singularity = np.zeros(dim).astype(bool)

    d = np.array([
        np.dot((x - b[i]),  V[i, :, 0])
        for i in range(dim)
        ])

    cond = (d * bias_sign(RD) >= 0)  # & (f > 0)

    Q = np.array([
        cond[i] * A[i, :, :] -
        ~cond[i] * np.real(sla.sqrtm(A[i, :, :].dot(A[i, :, :].T)))
        for i in range(dim)
        ])

    f = np.array([  # (x-b)A(x-b)^T - 1
        (np.dot((x - b[i]), np.dot(Q[i, :, :], (x.T - b[i].T))))
        for i in range(dim)
        ]) - 1 + singularity.astype(int)
    q = np.array([  # 2 * (x-b)A
        2 * np.dot(x - b[i], Q[i, :, :])
        for i in range(dim)
        ])
    H = Q

    scaling = (RD)**4/16
    scaling[singularity] = 1
    scaling = scaling * 1e-6

    if mode == -3:
        Ret = (H.swapaxes(0, 2) * np.sqrt(scaling)).swapaxes(0, 2)
    elif mode == -2:
        Ret = (q.T * np.sqrt(scaling)).T
    elif mode == -1:
        Ret = f * np.sqrt(scaling)
    elif mode == 0:
        Ret = 0.5 * np.sum(scaling * f**2)
    elif mode == 1:
        Ret = np.sum(scaling * (f * q.T), axis=1)
    elif mode == 2:
        inter = np.array([
            f[i] * H[i, :, :]
            + 2 * np.matrix(q[i]).T @ np.matrix(q[i])
            for i in range(dim)
            ])
        Ret = 2 * np.sum(
            (scaling * inter.swapaxes(0, 2)).swapaxes(0, 2), axis=0)

    return Ret


def GenMeasurements(N, n, Rs):

    # ### pre-process data ###
    # mapping of the stations to the differences
    mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])

    # range differences RD (equivalent to TDOA)
    RD = (- Rs[mp[:, 1]] + Rs[mp[:, 0]])  # meters

    # vectors and ranges between stations
    R = (N[mp[:, 1]] - N[mp[:, 0]])
    Rn = la.norm(R, axis=1)

    # Scaled measurements
    RD_sc = RD/Rn

    # ###determine usable measurements
    # lambda >0.99
    lamb_idx = np.where(abs(RD_sc) > 0.99)[0].astype(int)
    lamb_idx_N = mp[lamb_idx]
    z = np.zeros([len(lamb_idx), 3])
    z[:, 0] = lamb_idx
    z[:, 1:] = lamb_idx_N
    bad_stations_lamb = []
    while True:
        sta, freq = scist.mode(z[:, 1:], axis=None, nan_policy='omit')
        if freq > 1:
            rem_rows = (z[:, 1:] == sta[0]).any(axis=1)
            z[rem_rows, 1:] = np.nan
            bad_stations_lamb.append(int(sta[0]))
        else:
            break

    bad_meas_lamb = np.in1d(mp, bad_stations_lamb)\
        .reshape(len(mp), 2).any(axis=1)

    # Rn < 10km --> rewrite to remove station with least remaining measremnts
    prox_idx = np.where(Rn < 1e4)[0].astype(int)
    prox_idx_N = mp[prox_idx]
    bad_meas_prox = np.in1d(mp, prox_idx_N[:, 0])\
        .reshape(len(mp), 2).any(axis=1)

    # combine
    m_use = ~bad_meas_prox & ~bad_meas_lamb
    m_use[lamb_idx] = False

    # alternative: only discard lambda > 0.99
    # m_use = abs(RD_sc) < 0.99

    # alternative: just use all
    # m_use = np.ones(len(mp)).astype(bool)

    # error if not enough stations left
    Kmin = 1
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


def genx0(N, mp, RD_sc, h_baro):
    # find index of most "central" measurement (least difference of Range
    # difference relative to station distance)
    x0_idx = np.where(abs(RD_sc) == np.min(abs(RD_sc)))[0][0]

    # find the x0 as the position on the line between those 2 stations that
    # also satisfies the hyperbola
    x0 = N[mp[x0_idx, 0], :] + \
        (0.5 + RD_sc[x0_idx] / 2) * (N[mp[x0_idx, 1], :] - N[mp[x0_idx, 0], :])

    # if h_baro is not np.nan:
    if True:
        std_mean = np.mean(N, axis=0)
        x0 = std_mean / la.norm(std_mean) * (R1 + h_baro)

    return x0


def solve(N, n, Rs, h_baro=np.nan, x0=None):

    # ### check quality of measurements and discard accordingly
    # scale measurement to distance between stations
    mp, RD, R, Rn, RD_sc = GenMeasurements(N, n, Rs)

    # determine problem size
    dim = len(mp)

    # ### calculate quadratic form
    A, V, D, b, singularity = getHyperbolic(N, mp, dim, RD, R, Rn)

    if h_baro is not np.nan:
        # use aultitude info --> define equality constraint
        cons = sciop.NonlinearConstraint(
            lambda x: WGS84(x, h_baro, mode=0), 0, 0,
            jac=lambda x: WGS84(x, h_baro, mode=1),
            hess=lambda x, __: WGS84(x, h_baro, mode=2),
            )
    else:
        cons = ()

    # ### generate x0
    if x0 is None:
        x0 = genx0(N, mp, RD_sc, h_baro)

    # ### solve and record
    xlist = []
    sol = sciop.minimize(
                    lambda x: FJsq(x, A, b, dim, V, RD, Rn, mode=0),
                    x0,
                    jac=lambda x: FJsq(x, A, b, dim, V, RD, Rn, mode=1),
                    hess=lambda x: 0.25*FJsq(x, A, b, dim, V, RD, Rn, mode=2),
                    method='SLSQP',
                    tol=1e-3,
                    constraints=cons,
                    options={'maxiter': 100,
                             # 'xtol': 0.1,
                             },
                    callback=lambda xk: xlist.append(xk),
                    )

    xn = sol.x
    opti = 0
    cost = sol.fun
    nfev = sol.nfev
    niter = sol.nit
    ecode = 0

    # build diagnostic struct
    inDict = {'A': A, 'b': b, 'V': V, 'D': D, 'dim': dim, 'RD': RD, 'xn': xn,
              'fun': lambda x, m: FJsq(x, A, b, dim, V, RD, Rn, mode=m),
              'xlist': xlist, 'ecode': ecode, 'mp': mp, 'Rn': Rn, 'sol': sol}

    return xn, opti, cost, nfev, niter, RD, inDict


def Pandas_Wrapper(MR, NR, idx, NR_c, solmode='2d'):
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

    pdcross = MR.loc[idx, :]

    # ### preprocess stations and measurements
    # get number of stations
    n = pdcross['M']

    # get station locations and convert to cartesian
    stations = np.array(pdcross['n'])
    N = SP2CART(NR.loc[stations, ['lat', 'long', 'geoAlt']].to_numpy())

    # ### get unix time stamps of stations
    Rs_corr = np.array([NR_c.NR_corr[i - 1][3] for i in stations])
    Rs = np.array(pdcross['ns']) * 1e-9 * C0 + Rs_corr  # meters

    # baro radius
    if (solmode == '2d') or (solmode == '2drc'):
        h_baro = pdcross['baroAlt']  # meters
    else:
        h_baro = np.nan

    # actually solve
    try:
        xn, opti, cost, nfev, niter, RD, inDict =\
            solve(N, n, Rs, h_baro=h_baro)

        if solmode == '2drc':
            xn, opti, cost, nfev, niter, RD, inDict =\
                solve(N, n, Rs, h_baro=np.nan, x0=xn)

        xn_sph = CART2SP(xn)

        MR.at[idx, "xn_sph_lat"] = xn_sph[0]
        MR.at[idx, "xn_sph_long"] = xn_sph[1]
        MR.at[idx, "xn_sph_alt"] = xn_sph[2]

        MR.at[idx, "dim"] = inDict['dim']

        MR.at[idx, "fval"] = cost
        MR.at[idx, "optimality"] = opti
        MR.at[idx, "nfev"] = nfev
        MR.at[idx, "niter"] = niter

        MR.at[idx, "MLAT_status"] = inDict['ecode']

    except MLATError as e:
        xn_sph = np.zeros(3)
        xn_sph[:] = np.nan
        MR.at[idx, "MLAT_status"] = e.code
        inDict = {}

    return xn_sph, inDict
