# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:53:53 2020

@author: Till
"""

import MLATlib as lib

TRA_temp['MLAT'] = False
TRA_temp.loc[TRA_temp['MLAT_status'] == 0, 'MLAT'] = True

TRA2 = TRA_temp.copy()
TRA2['score'] = np.nan
SOL2 = SOL_temp.copy()
acs = np.unique(TRA2.loc[SOL2.index, 'ac'])
for ac in tqdm(acs):
    aco = lib.filt.aircraft(TRA_temp, SOL_temp, ac)
    aco.Interp(usepnts='adaptive')
    SOL2.loc[aco.ids] = aco.SOLac
    TRA2.loc[aco.ids] = aco.TRAac
