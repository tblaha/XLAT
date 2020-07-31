# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:00:24 2020

@author: Till
"""

import MLATlib as lib

# TRA_temp, SOL_temp = lib.ml.PruneResults(TRA, SOL, usefun=disc)
TRA_temp, SOL_temp = lib.ml.PruneResults(TRA, SOL, prunefun=None)
