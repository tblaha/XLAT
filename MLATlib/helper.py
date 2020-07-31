# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:31:08 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la

from scipy import constants as spyc

# global G, R0, F2M, C0,\
#     cosd, sind, tand, arcsind, arccosd, arctan2d,\
#     X, Y, Z, SP2CART,\
#     LAT, LONG, H, CART2SP


#%% constants
# ---------------------------------------------------------------------------
# Earth
a = 6378137              # mean equatorial radius
f = 1 / 298.257223563    # flattening
e = np.sqrt(2*f - f**2)  # eccentricity
G = spyc.g               # standard gravity, m/s/s
R1 = a*(3 - f)/3         # mean earth radius, m

# Physics
C0 = spyc.c              # speed of light, m/s

# freedom units
F2M = 0.3048             # conversion between feet and meter, m/ft


#%% angle helpers
# ---------------------------------------------------------------------------
# yep, I'm actually a MATLAB basic b*tch


def cosd(x):
    return np.cos(np.pi/180 * x)


def sind(x):
    return np.sin(np.pi/180 * x)


def tand(x):
    return np.tan(np.pi/180 * x)


def arcsind(x):
    return 180/np.pi * np.arcsin(x)


def arccosd(x):
    return 180/np.pi * np.arccos(x)


def arctan2d(z, x):
    return 180/np.pi * np.arctan2(z, x)


#%% coordinate transformations
# ----------------------------------------------------------------------------
# https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

def WGS84(x, mode=0):
    # mode == 0: return distance of x to the WGS84 surface
    #               (approx 6ppm between +-50km from surface)
    # mode == 1: return gradient vector pointing away from WGS84 surface
    # mode == 2: return Hessian
    A = np.diag(np.array([1/a,
                          1/a,
                          1/(a*(1 - f))
                          ])**2
                ) * R1
    
    if mode == 0:
        ret = 0.5 * (x @ A @ x - R1)
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
