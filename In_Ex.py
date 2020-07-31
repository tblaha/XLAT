# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:05:40 2020

@author: Till
"""


import scipy.spatial as ssp
import matplotlib.pyplot as plt
from matplotlib.path import Path

from tqdm import tqdm

NRnp = NR[['lat', 'long', 'geoAlt']].to_numpy()
# VALnp = VAL[['lat', 'long', 'geoAlt']].to_numpy()

i = 0
In = np.zeros(len(VAL)).astype(bool)
for idx, row in tqdm(VAL.iterrows(), total=len(VAL)):
    if TRA.loc[idx, 'M'] > 2:
    
        N = NRnp[np.array(TRA.loc[idx, 'n']) - 1, :]
        
        try:
            x_sph_GT = VAL.loc[idx].to_numpy()
        except KeyError:
            x_sph_GT = row[['lat', 'long', 'geoAlt']]
            print('fuck')
        
        hull = ssp.ConvexHull(N[:, 0:2])
        
        hull_path = Path(N[hull.vertices, 0:2])
        
        In[i] = hull_path.contains_point(x_sph_GT[0:2])
    i += 1


TRA.loc[VAL.index, 'Interior'] = In
