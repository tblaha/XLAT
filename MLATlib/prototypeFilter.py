# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:54:21 2020

@author: Till
"""

from MLATlib.helper import SP2CART, CART2SP

TRA2 = TRA_temp.copy()
TRA2['score'] = np.nan
SOL2 = SOL_temp.copy()
acs = np.unique(TRA2.loc[SOL2.index, 'ac'])

F = np.eye(6)
Qp = np.diag(np.array([0, 0, 1, 1, 0.1, 0.1]))
Rp = np.diag(np.array([1000, 1000, 1, 1000, 1000, 1]))
H = np.eye(6)
# H = np.zeros([3, 6])
# H[0:3, 0:3] = np.eye(3)
            
for ac in tqdm(acs):
    idxs = TRA2.loc[(TRA2['ac'] == ac) & (TRA2['MLAT'])].index
    start = True
    for idx in idxs:
        if start:
            x = SP2CART(SOL2.loc[idx].to_numpy())
            P = np.zeros([6, 6])
            start = False
            continue
        
        F[0:3, 3:] = np.eye(3) * 2
        
        
RMSE, cov, nv = lib.out.twoErrorCalc(SOL2, VAL, RMSEnorm=2)

print(RMSE)
print(cov*100)