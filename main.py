# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:51 2020

@author: Till
"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

from   constants import *

import              numpy  as np
from   numpy import linalg as la

import pandas as pd

import readLib   as rlib
import outLib    as olib 

MR = rlib.readMeasurements("../training_1_round_1/training_1_category_1/training_1_category_1.csv")
NR = rlib.readNodes("../training_1_round_1/training_1_category_1/sensors.csv")
SR = rlib.readSolutions("../training_1_round_1_result/training_1_category_1_result/training_1_category_1_result.csv")

#olib.writeSolutions("../training_1_round_1_result/test_out.csv", SR)


