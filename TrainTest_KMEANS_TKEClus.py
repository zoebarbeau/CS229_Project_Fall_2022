# -*- coding: utf-8 -*-

## ------------------------------------------------------------------------  ##
"""
Objective: train and test

Date: 10/24

To Do:

"""
## ------------------------------------------------------------------------  ##
" Load Packages "

#general
import numpy as np
import pandas as pd
import timeit as time
import os
import pickle
    
#plotting 
import matplotlib.pyplot as plt

# random forset
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# KMeans
from sklearn.cluster import KMeans

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
from utilities.plot_triangles import *
from utilities.plot_triangles_kmeans_TKE import *
from utilities.plot_triangles_clusters_TKE import *
from utilities.convert_DUCT import *
from utilities.time_print import *
from utilities.plot_overall_errors_cluster import *

time_start = time.default_timer()
time_stop = time_start*1.0
## ------------------------------------------------------------------------  ##
" Intialize "
Thres_Training_flag = 1
Thres_Testing_flag = 1

labels = ['bay_x','bay_y']

# FEATURE SETS
feature_set = {}

feature_set['FS_RF'] =  ['Ux', 'Uy', 'k', 'omega', 'p', 'wallDistance',
        'nut', 'epsilon', 'k_norm_u','Q_Crit', 'wall_Re','P_epsilon', 'dimless_shear', 'marker_shear']

# feature_set['FS2'] =  ['tr_Sij', 'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2']

# feature_set['FS3'] =  ['Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
#    'P_epsilon', 'dimless_shear', 'marker_shear']
    
# feature_set['FS4'] =  ['tr_Sij',
#    'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2',
#    'Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
#    'P_epsilon', 'dimless_shear', 'marker_shear']

## ------------------------------------------------------------------------  ##
" Choose cases "
cases_train = [
             #  'DUCT_1250',
             #  'DUCT_1100',
             #  'DUCT_1150',
             #  'DUCT_1250',
             #  'DUCT_1300',
             #  'DUCT_1350',
             #  'DUCT_1400',
             #  'DUCT_1500',
             #  'DUCT_1600',
             #  'DUCT_1800',
             #  'DUCT_2000',
             #  'DUCT_2205',
             #  'DUCT_2400',
             #  'DUCT_2600',
             #  'DUCT_2900',
             #  'DUCT_3200',
             #  'DUCT_3500',
             #  'PHLL_case_1p0',
             #  'PHLL_case_1p5',
             #  'BUMP_h20',
             #  'BUMP_h26',
             # 'BUMP_h31',
             # 'BUMP_h38',
             #  'BUMP_h42',
              'CNDV_20580']

cases_test = [
           'CNDV_12600']
           # 'PHLL_case_1p2',
           # 'PHLL_case_0p5',
           # 'PHLL_case_0p8']

DS_name = 'TESTCS229'
trail = '_trial2'
output_main = DS_name + trail
## ------------------------------------------------------------------------  ##
""" if this the first time using this data setup (training/test cases) + labels
    Then we need to create the different normalized cases
"""
if 1 == 1:
     create_norm_dict(cases_train,cases_test,labels,DS_name)
     time_stop =  time_print(time_start,time_stop,'Created Dictionaries')

## ------------------------------------------------------------------------  ##
" Load Data "

#norms = ['NormNone','Norm01','NormRe','NormNu']
norms = ['Norm01']
errortrack = {}
statetrack = {}

#Flags for PCA
PCA_flag = 0
numpca = 0
coef_v = []

for f in feature_set.keys(): # LOOP OVER FEATURE SETS 
    features = feature_set[f]
       
    for n in norms:# LOOP OVER NORMS SETS 
        print('---- STARTING Feature Set ',f, ' and normalization ', n, ' ----')
        features = feature_set[f]
      
        features_train_nc, _, labels_train_nc, thrss_train_nc = load_data_NN(cases_train,labels,features,n,DS_name)  
        features_test_nc,  _,  labels_test_nc,  thrss_test_nc = load_data_NN(cases_test,labels,features,n,DS_name)
        kmeans_path = 'ML_models/' + output_main +'/' + f + '/' + n + '/'
        kmeans_file = 'kmodel.pkl'
        
        if not os.path.exists(kmeans_path):
            os.makedirs(kmeans_path)
           

        numcluster = 5
        kmeans = KMeans(n_clusters=numcluster).fit(thrss_train_nc)
        prediction = kmeans.predict(thrss_test_nc)          
        pickle.dump(kmeans, open(kmeans_path+kmeans_file, 'wb'))
        
        #Plot clustering for each case
        for case in cases_test:
            SW_emphasis = 1
            df = pd.read_pickle('data/RANS_' + DS_name +'/'+ n +'/'+case+'_DICT.pkl') 
            thrs_test        = np.array(df['thres']) * SW_emphasis
            predict = kmeans.predict(thrs_test.reshape(-1,1))
            # get org RANS data
            filename = case + '_'
            # LES_true_path='\data\LES_true'
            if case.split('_')[0] == 'DUCT':
                RANS_path='\data\RANS'
                filetype = ".npy"
            else:
                RANS_path='\data\RANSv2'
                filetype = ".dat"
            RANS_org = load_dataset(RANS_path,filename, filetype, LES_true=False, LES = False, RANS = True)

            x = RANS_org['Cx']
            y = RANS_org['Cy']
            plt.figure(2)
            plt.scatter(x,y,c=predict)
            plt.savefig(kmeans_path + case + '_justclus' +'.png')
            plt.close()
       
        #use PCA
        if PCA_flag:
            features_train_nc, numpca = pca_features(features_train_nc,kmeans_path)
            features_test_nc = pca_features_test(features_test_nc,numpca,kmeans_path)
                
        #Train ML for each cluster
        for i in range(numcluster):
                      
            features_train = features_train_nc[kmeans.labels_ == i, :]
            labels_train  = labels_train_nc[kmeans.labels_ == i, :]
            thrss_train   = thrss_train_nc[kmeans.labels_ == i, :]
            features_test = features_test_nc[prediction == i, :] 
            labels_test  = labels_test_nc[prediction == i, :]
            thrss_test   = thrss_test_nc[prediction == i, :]
        
            if features_train.shape[0] == 0:
                continue
       
            if features_test.shape[0] == 0:
                break
        #NN 
        ## ------------------------------------------------------------------------  ##
            " Train ML "
            ep = 300 #123 # number of epochs
            bs = 1024# batch_size
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
               keras.layers.Reshape(target_shape=(features_test.shape[1],), input_shape=(features_test.shape[1],)),
               keras.layers.Dense(units=16, activation='relu',kernel_initializer=initializer),
               keras.layers.Dense(units=8, activation='relu',kernel_initializer=initializer),
               keras.layers.Dense(units=2, activation='linear',kernel_initializer=initializer)])
            
            #piecewise learning rate
            boundaries = [50, 200, 400, 600]
            values = [10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
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
            
                    
            outputpath = 'ML_models/' + output_main+ '/' + f + '/' + '/' + n + '/' + str(i) + '/'
            
            train_NN(model, bs, ep, outputpath,
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
for i in range(numcluster):
    kpath = 'ML_models/' + output_main+ '/' + f + '/' + '/' + n + '/' + str(i) + '/'
    et = pd.DataFrame(errortrack[kpath])
    st = pd.DataFrame(statetrack[kpath])
    et.to_pickle('ML_models/' + output_main+'/errors' + str(i) +'.pkl')
    st.to_pickle('ML_models/' + output_main+'/states' + str(i) +'.pkl')
    plot_overall_errors_clus(feature_set, norms,i, output_main)
## ------------------------------------------------------------------------  ##
# " Get sample triangle errors "

time_stop =  time_print(time_start,time_stop,'Load Testing')

## ------------------------------------------------------------------------  ##
" do complete error plots "
cols = {}
cols['DUCT'] = [19,33,45]
cols['CNDV'] = [300,400,550]
cols['BUMP'] = [200,300,385]
cols['PHLL'] = [88,62,8]


cases =  ['CNDV_12600']

colorstyle = 'wallD' # 'weight' 'none'
cmap_list = ['b']    # 'k', 'darkviolet', 'cadetblue', 'dimgray', 'fuchsia', 'midnightblue', 'purple', 'slateblue', 'indigo']

plot_triangle_kmeans_TKE(model, labels, feature_set, norms, 
                  DS_name, output_main, kmeans_file,numcluster,coef_v,numpca,PCA_flag,cases, cols,colorstyle,cmap_list,SW_emphasis)


plot_triangle_clusters_TKE(model, labels, feature_set, norms, 
                 DS_name, output_main, kmeans_file,numcluster,coef_v,numpca,PCA_flag,cases, cols,colorstyle,cmap_list,SW_emphasis)                