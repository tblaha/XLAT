# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:31:08 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la

from scipy import constants as spyc
from astropy import constants as astc

global G, R0, F2M, C0,\
    cosd, sind, tand, arcsind, arccosd, arctan2d,\
    X, Y, Z, SP2CART,\
    LAT, LONG, H, CART2SP

# constants:
# ---------------------------------------------------------------------------
C0 = spyc.c              # speed of light, m/s
G = spyc.g               # standard gravity, m/s/s
R0 = astc.R_earth.value  # earth mean equatorial radius, m

F2M = 0.3048             # conversion between feet and meter, m/ft


# angle helpers
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


# coordinate transformations
# ----------------------------------------------------------------------------
# https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions


def X(lat, long, h):
    return (h + R0) * sind(90 - lat) * cosd(long)


def Y(lat, long, h):
    return (h + R0) * sind(90 - lat) * sind(long)


def Z(lat, long, h):
    return (h + R0) * cosd(90 - lat)


def SP2CART(lat, long, h):
    return np.array([X(lat, long, h),
                     Y(lat, long, h),
                     Z(lat, long, h)
                     ])


def LAT(x, y, z):
    # using the tan formula for robustness around centre of earth
    return 90 - arctan2d(np.sqrt(x**2 + y**2), z)


def LONG(x, y, z):
    return arctan2d(y, x)


def H(x, y, z):
    return (la.norm(np.array([x, y, z]), axis=0) - R0)


def CART2SP(x, y, z):
    return np.array([LAT(x, y, z),
                     LONG(x, y, z),
                     H(x, y, z)
                     ])
