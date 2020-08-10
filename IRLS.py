# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:17:03 2020

@author: Till
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from MLATlib.helper import C0

R = np.array([50000,
              50000,
              50000,
              50000
              ])

tn = R / C0 * (np.array([1,
                         1,
                         1,
                         1.1
                         ]) 
               + np.random.random(4)/1000
               )

A = np.ones(4)

W = np.diag(np.ones(4)/2)

for i in range(10):
    y = sla.sqrtm(W) @ (R/C0 + tn)
    
    tac = 1/(A.T @ sla.sqrtm(W) @ A) * A.T @ y
    
    yres = sla.sqrtm(W) @ A*tac - y
    
    w = np.diag(W) * 1/yres**2
    W = np.diag(w/la.norm(w))



