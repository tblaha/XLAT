# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:40:34 2020

@author: Till
"""


import numpy as np
import numpy.linalg as la

import geopy as gp

a = 6378137
f = 1 / 298.257223563
e = np.sqrt(2*f - f**2)

# https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

def WGS84(x, mode=0):
    # mode == 0: return (approx ppm ~ 3000) distance of x to the WGS84
    # mode == 1: return gradient vector pointing away from WGS84 surface
    # mode == 2: return Hessian
    A = np.diag(np.array([1/a,
                          1/a,
                          1/(a*(1 - f))
                          ])**2
                ) * a
    
    if mode == 0:
        ret = 0.5 * (x @ A @ x - a)
    elif mode == 1:
        ret = (A @ x)
    elif mode == 2:
        ret = A
        
    return ret

def N(θ):
    # return the prime vertical radius of the ellipsoid along the circle 
    # defined by geodetic latitude θ
    return a / np.sqrt(1 - (e * sind(θ))**2)


def SP2CART(θφh):
    θφh = θφh.astype(float)
    
    if θφh.ndim == 1:
        θφh = np.array([θφh])
    
    # convert spherical to ECEF assuming geodetic latitude and WGS84
    θ, φ, h = θφh.T
    
    # prime vertical radius
    pvr = N(θ)
    
    # height (the 1 - e**2 factor compensates for the additional length of the
    # pvr that extends below the equator)
    z = ((1 - e**2) * pvr + h) * sind(θ)
    
    # magnitude of the vector from Center of Earth to 
    # projection onto the equatorial plane
    zp = (pvr + h) * cosd(θ)
    
    # decomposition onto x and y
    x = zp * cosd(φ)
    y = zp * sind(φ)
    
    # output
    out = np.array([x, y, z]).T
    return out


def CART2SP(xyz):
    xyz = xyz.astype(float)
    
    if xyz.ndim == 1:
        xyz = np.array([xyz])
    
    # convert ECEF to spherical with geodetic latitude and WGS84
    x, y, z = xyz.T
    
    # longitude (easy doing)
    φ = arctan2d(y, x)    
    
    # magnitude of the vector from Center of Earth to 
    # projection onto the equatorial plane
    zp = np.sqrt(x**2 + y**2)
    
    # Newton-Raphson iterations
    κ0 = (1 - e**2)**(-1) * np.ones(len(xyz))
    κ = κ0
    for i in range(2):
        c = (zp**2 + (1 - e**2) * z**2 * κ**2)**(3/2) / (a*e**2)
        κ = 1 + (zp**2 + (1 - e**2) * z**2 * κ**3) / (c - zp**2)
    
    # resolve the coordinate transformation κ = zp/z * tan(θ)
    θ = arctan2d(κ * z, zp)
    
    # get altitude normal above the WGS84
    h = e**(-2) * (κ**(-1) - κ0**(-1)) * np.sqrt(zp**2 + z**2 * κ**2)
    
    # output
    out = np.array([θ, φ, h]).T
    return out


# CART2SP(np.array([[2908061, 3465692, 4524139]]))