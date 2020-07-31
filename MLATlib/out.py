# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:26:06 2020

@author: Till
"""

from .helper import SP2CART

import numpy as np
from numpy import linalg as la
from geopy.distance import great_circle as gc


def twoErrorCalc(x, z, RMSEnorm=2):
    """
    2 dimensional RMSE using great circle distance on the ground truth height

    Parameters
    ----------
    x : pd.DataFrame
        Generated solution dataset (validation set).
    z : pd.DataFrame
        Ground truth solution dataset (validation set).
    RMSEnorm : scalar, optional
        RMSE root to use. The default is 2.

    Returns
    -------
    e : scalar
        RMSE.

    """

    # find the common indices (computed into x and preset in validation set z)
    sol_idx_bool = np.in1d(x.index, z.index)
    N = len(z.index)

    # get lat and longs and ground truth geo height
    lat_x, long_x = \
        np.array(x.loc[sol_idx_bool, ['lat', 'long']]).T
    lat_z, long_z, h_z = \
        np.array(z.loc[sol_idx_bool, ['lat', 'long', 'geoAlt']]).T

    # compute great circle distances ("2d" error) between guess and truth
    norm_vec = np.zeros(N)
    for i in range(N):
        try:
            norm_vec[i] = gc((lat_x[i], long_x[i]),
                             (lat_z[i], long_z[i])).meters
            if np.isnan(norm_vec[i]):
                norm_vec[i] = 0
        except ValueError:
            norm_vec[i] = 0

    if (norm_vec > 0).any():
        # only use lower 99th percentile (reverse engineered)
        p98 = np.percentile(norm_vec[norm_vec > 0], 98)
        nv_use = (norm_vec <= p98) & (norm_vec > 0)
    
        # RMSE error sum
        e = (np.sum(norm_vec[nv_use]**RMSEnorm)/sum(nv_use))**(1/RMSEnorm)
        
        # coverage
        cov = sum(nv_use) / len(z.index)
    
    else:
        e = np.nan
        cov = 0.98 * sum(~np.isnan(lat_x)) / len(z.index)

    return e, cov, norm_vec


def threeErrorCalc(x, z, RMSEnorm=2, pnorm=2):
    """
    3 dimensional RMSE using pnorm on cartesian coordinates.

    Parameters
    ----------
    x : pd.DataFrame
        Generated solution dataset (validation set).
    z : pd.DataFrame
        Ground truth solution dataset (validation set).
    RMSEnorm : scalar, optional
        RMSE root to use. The default is 2.
    pnorm : scalar, optional
        pnorm for the cartesian distance calculation. The default is 2.

    Returns
    -------
    e : scalar
        RMSE.

    """

    # find the common indices (computed into x and preset in validation set z)
    sol_idx_bool = np.in1d(x.index, z.index)

    # get lat and longs and ground truth geo height
    x_sph = np.array(x.loc[sol_idx_bool, ['lat', 'long', 'geoAlt']]
                     ).to_numpy()

    z_sph = z.loc[sol_idx_bool, ['lat', 'long', 'geoAlt']].to_numpy()

    # convert to cartesian
    cart_x = SP2CART(x_sph)
    cart_z = SP2CART(z_sph)

    # compute norm ("3d" error) between estimate and truth
    norm_vec = la.norm(cart_z - cart_x, pnorm, 1)
    
    if (norm_vec > 0).any():
        # only use lower 99th percentile (reverse engineered)
        p98 = np.percentile(norm_vec[norm_vec > 0], 98)
        nv_use = (norm_vec <= p98) & (norm_vec > 0)
    
        # RMSE error sum
        e = (np.sum(norm_vec[nv_use]**RMSEnorm)/sum(nv_use))**(1/RMSEnorm)
    
        # coverage
        cov = sum(nv_use) / len(z.index)
    
    else:
        e = np.nan
        cov = 0.98 * sum(~np.isnan(lat_x)) / len(z.index)

    return e, cov, norm_vec


def writeSolutions(filename, z):
    """
    write solution DataFrame to csv.

    Parameters
    ----------
    filename : string
        DESCRIPTION.
    z : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    zz = z.copy()
    zz.columns = ['latitude', 'longitude', 'geoAltitude']
    zz.to_csv(filename, index=True, index_label='id',
              na_rep='NaN')  # not-a-number string

    return 0
