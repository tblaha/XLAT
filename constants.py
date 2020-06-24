# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:31:08 2020

@author: Till
"""

import numpy as np

global G, R0, F2M, C0, cosd, sind, X, Y, Z

G = 9.81 # m/s/s
R0 = 6.371e+6 # m
F2M = 0.3048 # m/ft
C0  = 3e8 # m/s speed of light

cosd = lambda x: np.cos(np.pi/180 * x)
sind = lambda x: np.sin(np.pi/180 * x)

X = lambda lat, long, h: (h*F2M + R0) * sind(90-lat) * cosd(long)
Y = lambda lat, long, h: (h*F2M + R0) * sind(90-lat) * sind(long)
Z = lambda lat, long, h: (h*F2M + R0) * cosd(90-lat)
