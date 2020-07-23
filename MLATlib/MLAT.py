# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:40:56 2020

@author: Till
"""

from .helper import C0, R0, X, Y, Z, SP2CART, CART2SP

import numpy as np
import numpy.linalg as la
import scipy.optimize as sciop


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
        TDOA measurements. The first column is the subtrahend, second column is
        the minuend.
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
    fval = np.zeros([n + 1, 1])
    fval[-1] = la.norm(x)  # altitude

    # iterate over the items
    for i in np.arange(n):
        # Minuend (second column)  - Subtractor (first column)
        fval[i] = la.norm(N[(mp[i, 1]), :] - x)\
                  - la.norm(N[(mp[i, 0]), :] - x)

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
        array[2, :]: rho''(e) = diag(Del**2 * rho(e)) (el. seconds derivatives)
    """

    A = np.empty([3, e.size])
    scale = np.concatenate((np.ones([e.size - 1]), [beta]))
    A[0] = e * scale
    A[1] = scale  # derivatives wrt elements of e
    A[2] = np.zeros_like(e)  # seconds derivatives

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

    D = np.array([(x0[0] - n[0]), (x0[1] - n[1]), (x0[2] - n[2])])
    nD = la.norm(D)

    d = np.array([D[0], D[1], D[2]]) / nD

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
        TDOA measurements. The first column is the subtrahend, second column is
        the minuend.
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
        J[i, :] = delta(N[(mp[i, 1]), :], x0)\
                  - delta(N[(mp[i, 0]), :], x0)

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
    global X, Y, Z, SP2CART, CART2SP, C0

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

    # ### convert unix time stamps of stations to TDOA*C0 differential ranges
    # grab station TOA
    secs = np.array(MR.at[idx, 'ns']) * 1e-9  # get nanoseconds into seconds
    Rs = secs * C0  # meters

    # pre alloc
    b = np.zeros([dim + 1])
    mut_dists = np.zeros([dim, 1])
    """ Mapping matrix, maps a time difference index to the indices of the
    subtrahendand minuend.
    """
    mp = np.zeros([dim, 2])

    # iterate over the possible differences between 2 stations out of n
    mp = np.array([[i, j] for i in range(n) for j in range(i+1, n)])
    b[:-1] = secs[mp[:, 1]] - secs[mp[:, 0]]
    mut_dists = la.norm(N[mp[:, 1]] - N[mp[:, 0]])

    # add altitude (radius) target from baroAlt (light speed normalized)
    b[-1] = (MR.loc[idx, 'baroAlt'] + R0) / C0

    # ### assess quality of measurements

    # scaled TDOA ranges by distances between stations
    mut_dists_sc = b[0:-1] * C0 / mut_dists

    # select the K ones with the lowest ("sort") number --> most planar
    # surfaces as opposed to hyperboloids. This is a bool-array
    K = 2
    active_pnts = np.in1d(np.arange(dim), np.argsort(abs(mut_dists_sc).T)[:K])

    # abs(mut_dists_sc) > 1 are unphysical (and most likely unsolvable, too)
    if np.max(abs(mut_dists_sc[active_pnts])) > 1:
        raise FeasibilityError("Best measurements contain \
                               unphyiscal (lambda > 1) points")

    # ### actual solution process

    # generate initial guess --> find most suitable 2 stations to put the
    # initial guess in between. This satisfies at least 1 differential range
    # equation and should help convergences. Suitable means smallest TDOA
    # small comparted to distance between the stations (means hyperbolic
    # surfaces become more planar)
    x0_idx = np.where(abs(mut_dists_sc) == np.min(abs(mut_dists_sc)))[0][0]
    x0 = N[mp[x0_idx, 0], :]\
        + ((0.5 + mut_dists_sc[x0_idx] / 2)
           * (N[mp[x0_idx, 1], :] - N[mp[x0_idx, 0], :])
           )

    if solmode == 0:
        # Old fashioned NLLS
        itermax = 20  # max iterations

        it = 0
        xnminus1 = np.array([1, 1, 1]) * 1e10
        xn = x0
        while it < itermax and la.norm(xn-xnminus1) > 1:
            xnminus1 = xn
            xn = iterx(N, b, xnminus1)

            it = it + 1

        if la.norm(xn - xnminus1) > 1:
            raise ConvergenceError("Error norm not below threshold")

    elif solmode == 1:
        llsq_active = np.concatenate((active_pnts, [True]))
        # use scipy's LSQ solver based on Levenberg-Marqart with custom
        # Jacobian only solve at active_points
        sol = sciop.least_squares(
            lambda x: fun(N, mp, x)[llsq_active].T[0] - b[llsq_active] * C0,
            x0,
            jac=lambda x: np.array(Jac(N, mp, x)[llsq_active]),
            # method = 'dogbox', x_scale = 'jac',
            # loss = lambda x: rho_alt(x, 1e0), f_scale = 1e4
            # tr_solver = 'exact', \
            method='lm', x_scale='jac',
            max_nfev=200, xtol=1e-8, gtol=1e-8,
            verbose=0)
        xn = sol.x

        # this selects to used stations and adds it to the dataframe
        if 'n_used' not in MR:
            MR['n_used'] = ""
            MR['n_used'] = MR['n_used'].astype(object)

        try:
            MR.at[idx, 'n_used'] = \
                tuple(np.array(MR.at[idx, 'n'])[np.unique(mp[active_pnts])])
        except IndexError:
            print("asdf")

        # add some more
        MR.at[idx, "cost"] = sol.cost
        MR.at[idx, "nfev"] = sol.nfev

        if not sol.success:  # or sol.cost > 2e6:
            raise ConvergenceError("Not happy with sciop.least_squares' deed")

    return xn,\
        CART2SP(xn[0], xn[1], xn[2]),\
        (fun(N, mp, xn)-b*C0)[llsq_active]
