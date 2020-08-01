# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:54:21 2020

@author: Till
"""

from MLATlib.helper import SP2CART, CART2SP
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R


TRA2 = TRA_temp.copy()
TRA2['score'] = np.nan
SOL2 = SOL_temp.copy()
acs = np.unique(TRA2.loc[SOL2.index, 'ac'])

F = np.eye(6)
Qp = np.diag(np.array([0, 0, 1, 1, 0.1, 0.1]))
Rp = np.diag(np.array([10, 10, 1, 10, 10, 1]))
H = np.eye(6)
# H = np.zeros([3, 6])
# H[0:3, 0:3] = np.eye(3)
            
for ac in tqdm(acs):
    idxs = TRA2.loc[(TRA2['ac'] == ac) & (TRA2['MLAT'])].index
    start = 1
    for idx in idxs:
        if start > 0:
            xkm = np.zeros(6)
            xkm[0:3] = SP2CART(SOL2.loc[idx].to_numpy())
            Pkm = np.zeros([6, 6])
            tkm = TRA2.at[idx, 't']
            start -= 1
            continue
        
        
        tk = TRA2.at[idx, 't']
        Dt = tk - tkm
        if Dt < 1e-2:
            continue
        
        F[0:3, 3:] = np.eye(3) * Dt
        
        
        xkkm = F @ xkm
        
        zk = np.zeros_like(xkm)
        zk[0:3] = SP2CART(SOL2.loc[idx].to_numpy())
        zk[3:]  = (zk[0:3] - xkm[0:3]) / Dt
        
        Pkkm = F @ Pkm @ F.T + Qp * Dt
        
        yk = zk - H @ xkkm
        
        Sk = H @ Pkkm @ H.T + Rp * Dt
        
        Kk = Pkkm @ H.T @ la.inv(Sk)
        
        xk = xkkm + Kk @ yk
        
        
        
        tkm = tk
        xkm = xk
        Pkm = (np.eye(6) - Kk @ H) @ Pkkm
        
        
        
        SOL2.loc[idx, ['lat', 'long', 'geoAlt']] = CART2SP(xk[0:3])
        
        
        
        
RMSE, cov, nv = lib.out.twoErrorCalc(SOL2, VAL, RMSEnorm=2)

print(RMSE)
print(cov*100)