# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:40:03 2020

@author: Till
"""

from MLATlib.helper import R0, SP2CART
import MLATlib as lib
from geopy.distance import great_circle as gc
import numpy as np
from numpy import linalg as la

"""
with open('NR_c.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    [NR_c] = pickle.load(f)
"""

# seek_id = 1809908  # exterior: 12km
# seek_id = 1719111  # exterior: 40km
# seek_id = 1107342  # exterior: 5km
# seek_id = 971250   # exterior: 900m
seek_id = 1311466  # exterior: 650m
# seek_id = 1047204  # interior: 19m
# seek_id = 461011   # on boundary: 81m
# seek_id = 922043 # exterior: 305m
# seek_id = 1353230  # very ill-conditioned.. 232m
# seek_id = 329566  # exterior
# seek_id = 976425 # exterior 111m
# seek_id = 660619 # interior 2m
# seek_id = 1062583 # interior 33m

x_sph_GT = VAL.loc[seek_id].to_numpy()

# calculate clock correction
tserv = TRA.loc[seek_id, 't']

for i, row in enumerate(NR_c.NR_corr):
    for j, t in enumerate(row[0]):
        if t > tserv:
            NR_c.NR_corr[i][3] = row[2][j]
            break

# NR_c_sp = lib.sync.NR_corrector(TRA, NR, 0.1)
TRA.loc[seek_id, 'baroAlt'] = VAL.loc[seek_id, 'geoAlt']
# TRA.loc[seek_id, 'baroAlt'] = MR.loc[seek_id, 'baroAlt']

# start MLAT calculations
x_sph, inDict = lib.ml.NLLS_MLAT(TRA, NR, seek_id, NR_c, solmode='2d')

pp = lib.plot.HyperPlot(TRA, VAL, NR, seek_id, x_sph, inDict, SQfield=True)
pp.addPointByCoords(np.array([x_sph_GT[0:2]]))

print(la.norm(SP2CART(x_sph[0], x_sph[1], x_sph[2])
              - SP2CART(x_sph_GT[0], x_sph_GT[1], x_sph_GT[2])
              ))

xy_error = gc(tuple(x_sph[0:2]), tuple(x_sph_GT[0:2])).meters
   # * (R0 + x_sph_GT[2]) / R0

print(xy_error)

print(len(inDict['xlist']))
