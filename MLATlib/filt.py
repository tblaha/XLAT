# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:55:50 2020

@author: Till
"""

from .helper import SP2CART

import numpy as np
import numpy.linalg as la


class aircraft():
    def __init__(self, TRA, SOL, acid):
        self.ids = SOL.index.intersection(TRA.loc[TRA['ac'] == acid].index)
        self.TRAac = TRA.loc[self.ids]
        self.SOLac = SOL.loc[self.ids]
        self.TRAac['score'] = np.nan
        self.ids_nonan = self.ids[~np.isnan(self.SOLac['lat'])]
        
        self.recursive = False
        self.itercnt = 0
        
        self.eTH = 37

    def _InterpErrorEst(self, x_sph):
        x_MLAT = SP2CART(self.SOLac[['lat', 'long', 'geoAlt']].to_numpy())
        x_cart = SP2CART(x_sph)
        
        D = la.norm(x_MLAT - x_cart, axis=1)
        Dnonan = D[~np.isnan(D)]
        x_cart_nonan = x_cart[~np.isnan(x_cart[:, 0])]
        
        if len(Dnonan) == 0:
            return 0, Dnonan, x_cart_nonan
        
        Dfilt = Dnonan \
                - np.abs(np.convolve(Dnonan, np.array([-1, 2, -1]),
                                     mode='same'
                                     ))
        Dfilt[Dfilt < 0] = np.nan
        
        e = (np.nansum(Dfilt**2) / np.sum(~np.isnan(Dfilt)))**0.5
        
        return e, D, x_cart_nonan

    def Interp(self, usepnts='all'):
        
        if not self.recursive:
            # check if we even have data at all on the aircraft
            # if len(self.ids_nonan) == 0:
            #     return 
    
            # determine which points to use as nodes
            if usepnts == 'end':
                self.idsn = self.ids_nonan[[0, -1]]
                self.idsn_np = [0, -1]
            elif usepnts == 'all':
                self.idsn = self.ids_nonan
                self.idsn_np = range(len(self.idsn))
            elif usepnts == 'adaptive':
                self.recursive = True
                self.e_last = 1e9
                self.addn = []
                self.idsn = self.ids_nonan[[0, -1]]
                self.idsn_np = [0, -1]
            else:
                raise ValueError("usepnts must be one of \
                                 'end', \
                                 'all', \
                                 'adaptive'"
                                 )
        elif usepnts == 'adaptive':
            self.idsn_np = [0] + self.addn + [-1]
            self.idsn = self.ids_nonan[self.idsn_np]
            
        # all available time "t" if the aircraft
        ts = self.TRAac.loc[self.ids, 'MLATtime'].to_numpy()

        # get nodal values
        tn = self.TRAac.loc[self.idsn, 'MLATtime'].to_numpy()
        latn, longn, altn = self.SOLac.loc[self.idsn,
                                           ['lat', 'long', 'geoAlt']]\
            .to_numpy().T
        
        # interpolate the rest
        lats = np.interp(ts, tn, latn, left=np.nan, right=np.nan)
        longs = np.interp(ts, tn, longn, left=np.nan, right=np.nan)
        alts = np.interp(ts, tn, altn, left=np.nan, right=np.nan)
        x_sph = np.array([lats, longs, alts]).T
        
        # estimate accuracy
        self.e, self.D, self.x_cart_nonan = self._InterpErrorEst(x_sph)
        
        # !!! do not touch anymore !!!
        if self.recursive & (self.e > self.eTH):  # and (self.e_last - self.e) > 0.01:
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
            # score the values by curvature of the nodes
            x_nodes = self.x_cart_nonan[self.idsn_np]
            cos_score = np.ones(len(x_nodes))
            Diff = np.diff(x_nodes, axis=0)
            cos_score[1:-1] = np.sum(Diff[0:-1] * (Diff[1:]), axis=1)\
                / (la.norm(Diff[0:-1], axis=1)\
                   * la.norm(Diff[1:], axis=1)
                   )
            cos_score[0] = cos_score[1]
            cos_score[-1] = cos_score[-2]
            
            seg_score = (cos_score[0:-1] + cos_score[1:]) / 2
            
            scores = [[s] * len(
                self.ids[(self.ids >= self.idsn[i]) & (self.ids < self.idsn[i+1])]
                ) 
                for i, s in enumerate(seg_score)
                ]
            scores[-1].append(seg_score[-1])  # Off by 1 fix
            
            flat_scores = [item for sublist in scores for item in sublist]
                
            # assign cos score to _all points_ that were interpolated by the 
            # nodes in question --> avg somehow for the in-between nodes
            # after the fact, prune again by maximum cos score

            # write down
            self.SOLac[['lat', 'long', 'geoAlt']] = x_sph
            self.TRAac[['lat', 'long', 'geoAlt']] = x_sph
            
            # Flag the "true" MLAT nodes values used
            self.TRAac['MLAT'] = False
            self.TRAac.loc[self.idsn, 'MLAT'] = True
            
            # Flag the curvature score
            self.TRAac.loc[~np.isnan(self.TRAac['lat']), 'score'] = flat_scores
            
            return
        