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

def WGS84(x, h_baro, mode=0):
    # mode == 0: return distance of x to the hbaro-offset WGS84 surface
    #               (approx 6000ppm between +-50km from surface)
    # mode == 1: return gradient vector pointing outward
    # mode == 2: return Hessian
    A = np.diag(1 / (np.array([a,
                               a,
                               a*(1 - f)
                               ]) + h_baro)**2
                )
    
    if mode == 0:
        ret = 0.5 * R1 * (x @ A @ x - 1)
    elif mode == 1:
        ret = R1 * (A @ x)
    elif mode == 2:
        ret = R1 * A
        
    return ret

def N(θ):
    # return the prime vertical radius of the ellipsoid along the circle 
    # defined by geodetic latitude θ
    return a / np.sqrt(1 - (e * sind(θ))**2)


def pΕ(θ):
    # return the distance between any point on the circle with geodetic 
    # latitude θ and the centre of the earth
    pvr = N(θ)
    
    # z = ((1 - e**2) * N(θ)) * sind(θ)
    # zp = N(θ) * cosd(θ)
    # ret = np.sqrt(z**2 + zp**2)
    
    return pvr * np.sqrt(1 - e**2*sind(θ)**2)


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
    xyz = np.squeeze(np.array([x, y, z]).T)
    return xyz


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
    θφh = np.squeeze(np.array([θ, φ, h]).T)
    return θφh
