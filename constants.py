# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:31:08 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la

from scipy   import constants as spyc
from astropy import constants as astc


global G, R0, F2M, C0, cosd, sind, X, Y, Z, SP2CART, LAT, LONG, H, CART2SP


# constants:
G = spyc.g              # standard gravity, m/s/s
R0 = astc.R_earth.value # earth mean equatorial radius, m
F2M = 0.3048            # conversion between feet and meter, m/ft
C0  = spyc.c            # speed of light, m/s


# yep, I'm actually a MATLAB basic b*tch
cosd = lambda x: np.cos(np.pi/180 * x)
sind = lambda x: np.sin(np.pi/180 * x)
tand = lambda x: np.tan(np.pi/180 * x)
arcsind  = lambda x: 180/np.pi * np.arcsin(x)
arccosd  = lambda x: 180/np.pi * np.arccos(x)
arctan2d = lambda z,x: 180/np.pi * np.arctan2(z,x)


# coordinate transformations

#https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
X = lambda lat, long, h: (h*F2M + R0) * sind(90-lat) * cosd(long)
Y = lambda lat, long, h: (h*F2M + R0) * sind(90-lat) * sind(long)
Z = lambda lat, long, h: (h*F2M + R0) * cosd(90-lat)

SP2CART = lambda lat,long,h: np.array([X(lat,long,h), Y(lat,long,h), Z(lat,long,h)])


LAT  = lambda x,y,z: 90 - arctan2d(np.sqrt(x**2+y**2),z) # using the tan formula for robustness around centre of earth
LONG = lambda x,y,z: arctan2d(y,x)
H    = lambda x,y,z: (la.norm(np.array([x,y,z])) - R0)/F2M

CART2SP = lambda x,y,z: np.array([LAT(x,y,z), LONG(x,y,z), H(x,y,z)])

