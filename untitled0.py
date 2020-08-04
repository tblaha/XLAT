# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:48:34 2020

@author: Till
"""

import numpy.linalg as la

def genA(M):
    dim = lambda x: int(x*(x-1)/2)
    dimM = dim(M)
    
    A = np.zeros([dimM, M])
    while M > 1:
        M -= 1
        A[(dimM - dim(M+1)):(dimM - dim(M+1) + M), 0:M] += np.eye(M)
        A[(dimM - dim(M+1)):(dimM - dim(M+1) + M), -M:] -= np.eye(M)
    
    return A, dimM, la.matrix_rank(A)



