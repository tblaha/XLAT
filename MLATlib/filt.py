# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:55:50 2020

@author: Till
"""


import numpy as np
import numpy.linalg as la
from geopy.distance import great_circle as gc
from .helper import SP2CART


class aircraft():
    def __init__(self, TRA, SOL, acid):
        self.ids = SOL.index.intersection(TRA.loc[TRA['ac'] == acid].index)
        self.TRAac = TRA.loc[self.ids]
        self.SOLac = SOL.loc[self.ids]
        self.ids_nonan = self.ids[~np.isnan(self.SOLac['lat'])]
        
        self.recursive = False
        self.itercnt = 0

    def _InterpErrorEst(self, x_sph):
        x_MLAT = \
            SP2CART(self.SOLac['lat'], 
                    self.SOLac['long'], 
                    self.SOLac['geoAlt']
                    )
        x_cart = SP2CART(x_sph[:, 0],
                         x_sph[:, 1],
                         x_sph[:, 2]
                         )
        
        D = la.norm(x_MLAT - x_cart, axis=0)
        Dnonan = D[~np.isnan(D)]
        
        Dfilt = Dnonan \
                - np.abs(np.convolve(Dnonan, np.array([-1, 2, -1]),
                                     mode='same'
                                     ))
        Dfilt[Dfilt < 0] = np.nan
        
        e = (np.nansum(Dfilt**2) / np.sum(~np.isnan(Dfilt)))**0.5
        
        return e, D

    def Interp(self, usepnts='all'):
        
        if not self.recursive:
            # check if we even have data at all on the aircraft
            if len(self.ids_nonan) == 0:
                return 
    
            # determine which points to use as nodes
            if usepnts == 'end':
                idsn = self.ids_nonan[[0, -1]]
            elif usepnts == 'all':
                idsn = self.ids_nonan
            elif usepnts == 'adaptive':
                self.recursive = True
                self.e_last = 1e9
                self.addn = []
                idsn = self.ids_nonan[[0, -1]]
            else:
                raise ValueError("usepnts must be one of \
                                 'end', \
                                 'all', \
                                 'adaptive'"
                                 )
        elif usepnts == 'adaptive':
            idsn = self.ids_nonan[[0] + self.addn + [-1]]
            
        # all available time "t" if the aircraft
        ts = self.TRAac.loc[self.ids, 't'].to_numpy()

        # get nodal values
        tn = self.TRAac.loc[idsn, 't'].to_numpy()
        latn, longn, altn = self.SOLac.loc[idsn,
                                           ['lat', 'long', 'geoAlt']]\
            .to_numpy().T
        
        # interpolate the rest
        lats = np.interp(ts, tn, latn, left=np.nan, right=np.nan)
        longs = np.interp(ts, tn, longn, left=np.nan, right=np.nan)
        alts = np.interp(ts, tn, altn, left=np.nan, right=np.nan)
        x_sph = np.array([lats, longs, alts]).T
        
        # estimate accuracy
        self.e, self.D = self._InterpErrorEst(x_sph)
        
        if self.recursive & (self.e > 125):  # and (self.e_last - self.e) > 0.01:
            Dnonan = self.D[~np.isnan(self.D)]
            Dfilt = Dnonan \
                - np.abs(np.convolve(Dnonan, np.array([-1, 2, -1]),
                                     mode='same'
                                     ))
            
            self.addn.append(np.argmax(Dfilt))
            self.addn.sort()
            self.e_last = self.e
            self.itercnt += 1
            self.Interp(usepnts='adaptive')
            
            return
        else:
            # write down
            self.SOLac[['lat', 'long', 'geoAlt']] = x_sph
            self.TRAac[['lat', 'long', 'geoAlt']] = x_sph
            
            # Flag the "true" MLAT nodes values used
            self.TRAac['MLAT'] = False
            self.TRAac.loc[idsn, 'MLAT'] = True
            
            return
        


"""

    cur_id = TRA.loc[TRA['ac'] == ac].index
    cur_id = SOL2.index.intersection(cur_id)
    tempSOL = SOL2.loc[cur_id, 'long']
    cur_nonans = tempSOL.index[~np.isnan(tempSOL)]

    t = TRA.loc[cur_id, 't']
    t_nonan = t[cur_nonans].to_numpy()

    long = SOL.loc[cur_nonans, 'long'].to_numpy()
    lat = SOL.loc[cur_nonans, 'lat'].to_numpy()

    if len(lat):
        SOL2.at[cur_id, 'long'] = np.interp(t, t_nonan, long,
                                            left=np.nan, right=np.nan)
        SOL2.at[cur_id, 'lat'] = np.interp(t, t_nonan, lat,
                                           left=np.nan, right=np.nan)

        TRA.loc[cur_id, ['long', 'lat']] = SOL2.loc[cur_id, ['long', 'lat']]
"""