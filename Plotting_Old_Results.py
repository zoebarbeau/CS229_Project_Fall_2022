# -*- coding: utf-8 -*-

## ------------------------------------------------------------------------  ##
"""
Objective: 
Date: 11/29
To Do:

"""
## ------------------------------------------------------------------------  ##
" Load Packages "

#general
import numpy as np
import pandas as pd
import timeit as time
    
#plotting 
import matplotlib.pyplot as plt

# random forset
from sklearn.ensemble import RandomForestRegressor
# NN 
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)

# load dataset shape conversions 
from utilities.create_norm_dict import *
from utilities.load_data_NN import *
from utilities.train_NN import *
from utilities.compute_error_statistics import *
from utilities.plot_overall_errors import *
from utilities.plot_triangles_oldtrails import *

from utilities.time_print import *

time_start = time.default_timer()
time_stop = time_start*1.0
## ------------------------------------------------------------------------  ##
" Intialize "
Thres_Training_flag = 1
Thres_Testing_flag = 1

labels = ['bay_x','bay_y']

# FEATURE SETS
feature_set = {}
# feature_set['FS1'] =  ['Ux', 'Uy', 'Uz', 'k', 'omega', 'p', 'wallDistance',
#         'nut', 'epsilon', 'S_0,0', 'S_1,1', 'S_0,1', 'S_1,0',
#         'R_0,0', 'R_1,1', 'R_0,1', 'R_1,0', 'dk_0', 'dk_1', 'dp_0',
#         'dp_1']

# feature_set['FS2'] =  ['tr_Sij',
# 'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2',
# 'tr_Rij_2_Sij', 'tr_Rij_2_Sij_Rij_Sij_2']

# feature_set['FS3'] =  ['Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
# 'P_epsilon', 'dimless_shear', 'marker_shear']
    
feature_set['FS4'] =  ['tr_Sij',
'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2',
'tr_Rij_2_Sij', 'tr_Rij_2_Sij_Rij_Sij_2','Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
'P_epsilon', 'dimless_shear', 'marker_shear']



## ------------------------------------------------------------------------  ##
" Choose cases "
cases_train = ['BUMP_h20',
         'BUMP_h26',
         'BUMP_h38',
         'BUMP_h42']
cases_test = ['BUMP_h31']

DS_name = 'Trail1_Base'
trail = '_Bump_in'
output_main = DS_name + trail

## ------------------------------------------------------------------------  ##
" Load Data "

# norms = ['NormNone','Norm01','NormRe','NormNu']
norms = ['NormNone','NormRe','NormNu','Norm01']#['NormNone','Norm01']


## ------------------------------------------------------------------------  ##
" do complete error plots "
cols = {}
cols['DUCT'] = [19,33,45]
cols['CNDV'] = [300,400,550]
cols['BUMP'] = [200,300,385]
cols['PHLL'] = [88,62,8]

cases = ['BUMP_h20','BUMP_h31','BUMP_h42']
colorstyle = 'none' # 'weight' 'none'
for f in feature_set.keys(): 
    features = feature_set[f]
    initializer = tf.keras.initializers.HeNormal(seed=42)

    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(len(features),), input_shape=(len(features),)),
            keras.layers.Dense(units=8, activation='ReLU',kernel_initializer=initializer),
            keras.layers.Dense(units=6, activation='ReLU',kernel_initializer=initializer),
            keras.layers.Dense(units=4, activation='ReLU',kernel_initializer=initializer),
            keras.layers.Dense(units=2, activation='linear',kernel_initializer=initializer) ])
        
    plot_triangle_oldtrails(model, labels,  features, f, norms, 
                    'Bump_Inter', output_main, cases, cols,colorstyle)
                