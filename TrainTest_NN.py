# -*- coding: utf-8 -*-

## ------------------------------------------------------------------------  ##
"""
Objective: NN

2L, 16, 8
Leaky Relu

ep 300
bs 1024
learning rate 10^-2, 10^-3, 10^-4

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

# NN
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)

# load dataset shape conversions
from utilities.create_norm_dict import *
from utilities.load_data_NN import *
from utilities.train_NN_229 import *
from utilities.compute_error_statistics import *
from utilities.plot_overall_errors import *
from utilities.plot_triangles_v2 import *

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
feature_set['FS1'] =  ['Ux', 'Uy', 'Uz', 'k', 'omega', 'p', 'wallDistance',
        'nut', 'epsilon', 'S_0,0', 'S_1,1', 'S_0,1', 'S_1,0',
        'R_0,0', 'R_1,1', 'R_0,1', 'R_1,0', 'dk_0', 'dk_1', 'dp_0',
        'dp_1']

feature_set['FS2'] =  ['tr_Sij',
'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2',
'tr_Rij_2_Sij', 'tr_Rij_2_Sij_Rij_Sij_2']

feature_set['FS3'] =  ['Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
'P_epsilon', 'dimless_shear', 'marker_shear']

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

DS_name = 'Bump_Inter'
trail = '_CS229_NN2L_baseline'
output_main = DS_name + trail
## ------------------------------------------------------------------------  ##
""" if this the first time using this data setup (training/test cases) + labels
    Then we need to create the different normalized cases
"""
if 1 == 0:
    create_norm_dict(cases_train,cases_test,labels,DS_name)
    time_stop =  time_print(time_start,time_stop,'Created Normalized Dictionaries')

## ------------------------------------------------------------------------  ##
" Load Data "

norms = ['NormNone','Norm01','NormRe','NormNu']

errortrack = {}
statetrack = {}

for f in feature_set.keys(): # LOOP OVER FEATURE SETS
    features = feature_set[f]

    for n in norms:# LOOP OVER NORMS SETS
        print('---- STARTING Feature Set ',f, ' and normalization ', n, ' ----')

        features_train, _, labels_train, thrss_train = load_data_NN(cases_train,labels,features,n,DS_name)
        time_stop =  time_print(time_start,time_stop,'Load Training')

        features_test,  _,  labels_test,  thrss_test = load_data_NN(cases_test,labels,features,n,DS_name)
        time_stop =  time_print(time_start,time_stop,'Load Testing')

        #NN
        ## ------------------------------------------------------------------------  ##
        " Train ML "
        ep = 300 # number of epochs
        bs =  1024# batch_size
        SW_emphasis = 1

        train_num = np.shape(features_train)[0]
        test_num = np.shape(features_test)[0]

        train_order = np.linspace(0,train_num-1,train_num,dtype=int)
        np.random.shuffle(train_order)
        test_order = np.linspace(0,test_num-1,test_num,dtype=int)
        np.random.shuffle(test_order)

        # training data set
        features_train_tf = tf.cast(features_train[train_order,:], tf.float32)
        labels_train_tf = tf.cast(labels_train[train_order,:], tf.float32)
        labels_train_sf = labels_train[train_order,:]
        thrss_train_tf = tf.cast(thrss_train[train_order,:], tf.float32)
        thrss_train_sf = thrss_train[train_order,:]


        # validation / testing set
        features_test_tf = tf.cast(features_test[test_order[0:test_num//2]], tf.float32)
        labels_test_sf = labels_test[test_order[0:test_num//2]]
        thrss_test_sf = thrss_test[test_order[0:test_num//2]]

        features_valid_tf = tf.cast(features_test[test_order[test_num//2::]], tf.float32)
        labels_valid_sf = labels_test[test_order[test_num//2::]]
        thrss_valid_sf = thrss_test[test_order[test_num//2::]]

        # define NN
        initializer = tf.keras.initializers.HeNormal(seed=42)

        model = keras.Sequential([
            keras.layers.Reshape(target_shape=(len(features),), input_shape=(len(features),)),
            keras.layers.Dense(units=16, activation='relu',kernel_initializer=initializer),
            keras.layers.Dense(units=8, activation='relu',kernel_initializer=initializer),
            keras.layers.Dense(units=2, activation='linear',kernel_initializer=initializer) ])

        # decaying learning rates
        boundaries = [50, 200]
        values = [10**-2, 10**-3, 10**-4]
        lr_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
        opt = tf.keras.optimizers.Adam(lr_fn)

        # # abs error
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.mae,
                      metrics=['mae'])

        # get sample weights setup
        if Thres_Training_flag:
            SW_train = thrss_train_sf * SW_emphasis
            SW_train_NN = thrss_train_tf * SW_emphasis

        else:
            SW_train = 1.0
            SW_train_NN = tf.cast(np.ones(train_num), tf.float32)

        if Thres_Testing_flag:
            SW_test = thrss_test_sf * SW_emphasis
            SW_valid = thrss_valid_sf * SW_emphasis
        else:
            SW_test = 1.0
            SW_valid = 1.0


        outputpath = 'ML_models/' + output_main+ '/' + f + '/' + n + '/'

        train_NN_229(model, bs, ep, outputpath,
                 features_train_tf, labels_train_tf, SW_train_NN,
                 features_valid_tf, labels_valid_sf)
        time_stop =  time_print(time_start,time_stop,'Training Complete')

        ## ------------------------------------------------------------------------  ##
        " Get summary Errors "
        errortrack,statetrack = compute_error_statistics(errortrack, statetrack, outputpath, model,
                                     features_train_tf, labels_train_sf, SW_train,
                                     features_test_tf, labels_test_sf, SW_test,
                                     features_valid_tf, labels_valid_sf, SW_valid)
        time_stop =  time_print(time_start,time_stop,'Summary Errors')

print('--------- Done with all training ---------')
et = pd.DataFrame(errortrack)
st = pd.DataFrame(statetrack)
et.to_pickle('ML_models/' + output_main+'/errors.pkl')
st.to_pickle('ML_models/' + output_main+'/states.pkl')
## ------------------------------------------------------------------------  ##
" Get sample triangle errors "
plot_overall_errors(feature_set, norms, output_main)
time_stop =  time_print(time_start,time_stop,'Load Testing')

## ------------------------------------------------------------------------  ##
" do complete error plots "
cols = {}
cols['DUCT'] = [19,33,45]
cols['CNDV'] = [300,400,550]
cols['BUMP'] = [200,300,385]
cols['PHLL'] = [88,62,8]

cases = ['BUMP_h20','BUMP_h31','BUMP_h42']
colorstyle = 'weight' # 'weight' 'none'
for f in feature_set.keys():
    features = feature_set[f]

    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(len(features),), input_shape=(len(features),)),
            keras.layers.Dense(units=16, activation='relu',kernel_initializer=initializer),
            keras.layers.Dense(units=8, activation='relu',kernel_initializer=initializer),
            keras.layers.Dense(units=2, activation='linear',kernel_initializer=initializer) ])

    plot_triangle_v2(model, labels,  features, f, norms,
                    DS_name, output_main, cases, cols,colorstyle,SW_emphasis)
