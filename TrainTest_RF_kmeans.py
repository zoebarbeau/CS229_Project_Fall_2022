# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 21:23:30 2022

@author: zoeba
"""

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
    
#plotting 
import matplotlib.pyplot as plt
import time
# random forset
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# NN 
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)

# load dataset shape conversions 
from utilities.create_norm_dict import *
from utilities.load_data_NN import *
from utilities.train_NN import *
from utilities.compute_error_statistics_RF import *
from utilities.plot_triangles_kmeans_forest import *
from utilities.plot_triangles_clusters_forest import *
from utilities.compute_error_statistics_RF import *
from utilities.pca_features import *
from utilities.plot_overall_errors import *
from utilities.plot_overall_errors_cluster import *


#PCA
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# KMeans
from sklearn.cluster import KMeans



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
# feature_set['FS_All'] =  ['Ux', 'Uy', 'Uz', 'k', 'omega', 'p', 'wallDistance',
#         'nut', 'epsilon', 'S_0,0', 'S_1,1', 'S_0,1', 'S_1,0',
#         'R_0,0', 'R_1,1', 'R_0,1', 'R_1,0','tr_Sij', 'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 
#         'tr_Rij_2_Sij_2','Q_Crit', 'k_epsilon_norm',
#         'k_norm_u', 'wall_Re','P_epsilon', 'dimless_shear', 'marker_shear']

feature_set['FS_RF'] =  ['Ux', 'Uy', 'k', 'omega', 'p', 'wallDistance',
        'nut', 'epsilon', 'k_norm_u','Q_Crit', 'wall_Re','P_epsilon', 'dimless_shear', 'marker_shear']
    
    
# feature_set['FS1'] =  ['Ux', 'Uy', 'Uz', 'k', 'omega', 'p', 'wallDistance',
#         'nut', 'epsilon', 'S_0,0', 'S_1,1', 'S_0,1', 'S_1,0',
#         'R_0,0', 'R_1,1', 'R_0,1', 'R_1,0']

# feature_set['FS2'] =  ['tr_Sij', 'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2']

# feature_set['FS3'] =  ['Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
#    'P_epsilon', 'dimless_shear', 'marker_shear']
    
# feature_set['FS4'] =  ['tr_Sij',
#   'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2',
#   'Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
#   'P_epsilon', 'dimless_shear', 'marker_shear']
# ## ------------------------------------------------------------------------  ##
" Choose cases "
cases_train = [
             'DUCT_1250',
             'DUCT_1100',
             'DUCT_1150',
             'DUCT_1250',
             'DUCT_1300',
             'DUCT_1350',
             'DUCT_1400',
             'DUCT_1500',
             'DUCT_1600',
             'DUCT_1800',
             'DUCT_2000',
             'DUCT_2205',
             'DUCT_2400',
             'DUCT_2600',
             'DUCT_2900',
             'DUCT_3200',
             'DUCT_3500',
             'PHLL_case_1p0',
             'PHLL_case_1p5',
             'BUMP_h20',
             'BUMP_h26',
            'BUMP_h31',
             'BUMP_h38',
            'BUMP_h42',
            'CNDV_20580']
          # 'DUCT_1300',
          # 'DUCT_1350',
          # 'DUCT_1400',
          # 'DUCT_1600',
          # 'DUCT_1800',
          # 'DUCT_2000',
          # 'DUCT_2205',
          # 'DUCT_2600',
          # 'DUCT_2900',
          # 'DUCT_3200']

cases_test = [
          'CNDV_12600',
          'PHLL_case_1p2',
          'PHLL_case_0p5',
          'PHLL_case_0p8']

DS_name = 'TESTCS229'
trail = 'RandomForest_kmeans5_AllBump_2PH1p0_1p5_1CNDV_FRF_50tree_15depth_norm01__PCA2_TKE'
output_main = DS_name + trail
## ------------------------------------------------------------------------  ##
""" if this the first time using this data setup (training/test cases) + labels
    Then we need to create the different normalized cases
"""
if 1 == 1:
    create_norm_dict(cases_train,cases_test,labels,DS_name)
    time_stop =  time_print(time_start,time_stop,'Created Normalized Dictionaries')

## ------------------------------------------------------------------------  ##
" Load Data "

norms = ['Norm01']
#norms = ['NormNone']
errortrack = {}
statetrack = {}
PCA = 0
for f in feature_set.keys(): # LOOP OVER FEATURE SETS 
    features = feature_set[f]
    

    for n in norms:# LOOP OVER NORMS SETS 
        print('---- STARTING Feature Set ',f, ' and normalization ', n, ' ----')
           
        features_train_nc, _, labels_train_nc, thrss_train_nc = load_data_NN(cases_train,labels,features,n,DS_name)
        time_stop =  time_print(time_start,time_stop,'Load Training')

        features_test_nc,  _,  labels_test_nc,  thrss_test_nc = load_data_NN(cases_test,labels,features,n,DS_name)
        time_stop =  time_print(time_start,time_stop,'Load Testing')
        
        #RF
        ## ------------------------------------------------------------------------  ##
        #kmeans save path
        kmeans_path = 'ML_models/' + output_main +'/' + f + '/' + n + '/'
        kmeans_file = 'kmodel.pkl'
        
        if not os.path.exists(kmeans_path):
            os.makedirs(kmeans_path)
           
        coef_v = []
        numcluster = 5
        
        #kmeans predictions, right now using TKE
        kmeans = KMeans(n_clusters=numcluster).fit(thrss_train_nc)
        k_prediction = kmeans.predict(thrss_test_nc)          
        pickle.dump(kmeans, open(kmeans_path+kmeans_file, 'wb'))
        
        #For each cluster, make a forest
        for i in range(numcluster):
            numpca = 2
            
            feature_cluster = features_train_nc[kmeans.labels_ == i, :]
            feature_test_cluster = features_test_nc[k_prediction == i, :]
            #use PCA
            if PCA:
                feature_cluster = pca_features_test(feature_cluster,numpca,kmeans_path)
                feature_test_cluster = pca_features_test(feature_test_cluster,numpca,kmeans_path)
                    
            features_train = feature_cluster
            labels_train  = labels_train_nc[kmeans.labels_ == i, :]
            thrss_train   = thrss_train_nc[kmeans.labels_ == i, :]
            features_test = feature_test_cluster
            labels_test  = labels_test_nc[k_prediction == i, :]
            thrss_test   = thrss_test_nc[k_prediction == i, :]
        
            if features_train.shape[0] == 0:
                continue
       
            if features_test.shape[0] == 0:
                break
            
            ## RF fit
            forest = RandomForestRegressor(n_estimators=50,max_depth=15, random_state=0, n_jobs = -1)
            forest.fit(features_train,labels_train,sample_weight=thrss_train.reshape(-1,))
            forest_file = 'RFmodel.pkl'
            
            outputpath = 'ML_models/' + output_main+ '/' + f + '/' + n + '/' + str(i) + '/'
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)
                
            #Save RF
            pickle.dump(forest, open(outputpath+forest_file, 'wb'))
            
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
               
            #Make predictions
            prediction = forest.predict(features_test_tf)
            result = permutation_importance(forest, features_test_tf, labels_test_sf, n_repeats=10, random_state=42)
            score = forest.score(features_test_tf, labels_test_sf)
    
            #Plot permutation importances
            forest_importances = pd.Series(result.importances_mean)
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
            ax.set_title("Feature importances using permutation importance ")
            ax.set_ylabel("Mean accuracy decrease")
            fig.tight_layout()
            plt.show()
            fig.savefig(outputpath+'feature_importance_perm.png')
            plt.close()
            
            #Plot MDI importances
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
            fig.savefig(outputpath+'feature_importance_mdi.png')
            plt.show()
            plt.close()
            
            SW_emphasis = 1
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
                          
           
    
    
            ## ------------------------------------------------------------------------  ##
            " Get summary Errors "
            errortrack,statetrack = compute_error_statistics_RF(errortrack, statetrack, outputpath, forest_file,
                                         features_train_tf, labels_train_sf, SW_train,
                                         features_test_tf, labels_test_sf, SW_test,
                                         features_valid_tf, labels_valid_sf, SW_valid)     
            time_stop =  time_print(time_start,time_stop,'Summary Errors')

print('--------- Done with all training ---------')
for i in range(numcluster):
    kpath = 'ML_models/' + output_main+ '/' + f + '/' + n + '/' + str(i) + '/'
    et = pd.DataFrame(errortrack[kpath])
    st = pd.DataFrame(statetrack[kpath])
    et.to_pickle('ML_models/' + output_main+'/errors' + str(i) +'.pkl')
    st.to_pickle('ML_models/' + output_main+'/states' + str(i) +'.pkl')
    plot_overall_errors_clus(feature_set, norms,i, output_main)
    
time_stop =  time_print(time_start,time_stop,'Load Testing')

## ------------------------------------------------------------------------  ##
" do complete error plots "
cols = {}
cols['DUCT'] = [19,33,45]
cols['CNDV'] = [300,400,550]
cols['BUMP'] = [200,300,385]
cols['PHLL'] = [88,62,8]

cases = [  'CNDV_12600',
          'PHLL_case_1p2',
          'PHLL_case_0p5',
          'PHLL_case_0p8']
colorstyle = 'wallD' # 'weight' 'none'
cmap_list = ['b']
numpca=3
plot_triangle_kmeans_forest(labels, feature_set, norms, 
                  DS_name, output_main, forest_file, kmeans_file,numcluster,coef_v,numpca,PCA,cases, cols,colorstyle,cmap_list,SW_emphasis)


plot_triangle_clusters_forest(labels, feature_set, norms, 
                 DS_name, output_main, forest_file, kmeans_file,numcluster,coef_v,numpca,PCA,cases, cols,colorstyle,cmap_list,SW_emphasis)                
                