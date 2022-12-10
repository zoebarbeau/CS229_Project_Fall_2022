# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
features = ['FS4']
norms = ['None','01']


DS_name = 'CC_ALL'
trail = '_NNBaseline_CompareforObj4'
output_main = DS_name + trail
errors = pd.read_pickle('ML_models/'+output_main+'/errors.pkl')

all_mean_test = np.zeros((len(features)*len(norms),1))
all_mean_valid = np.zeros((len(features)*len(norms),1))
all_mean_train = np.zeros((len(features)*len(norms),1))
all_w_mean_test = np.zeros((len(features)*len(norms),1))
all_w_mean_valid = np.zeros((len(features)*len(norms),1))
all_w_mean_train = np.zeros((len(features)*len(norms),1))


for i,e in enumerate(errors):
    e_temp = errors[e]
    all_mean_test[i] = np.mean(e_temp[0:2])
    all_mean_valid[i] = np.mean(e_temp[2:4])
    all_mean_train[i] = np.mean(e_temp[4:6])
    all_w_mean_test[i] = np.mean(e_temp[18:20])
    all_w_mean_valid[i] = np.mean(e_temp[20:22])
    all_w_mean_train[i] = np.mean(e_temp[22:24])
        
print('overall test ERRORS ')

# for i in range(len(all_w_mean_train)):
#     print(all_w_mean_train[i,0],all_w_mean_valid[i,0])
    # if i % 2 != 0:
        # print(all_w_mean_valid[i,0])

for i in range(len(all_w_mean_train)):
    if i%2 !=0:
        print(all_w_mean_test[i,0])
# for i in all_mean_train:
#     print(i[0])
# print('----------')
# for i in all_mean_valid:
#     print(i[0])
# print('-------weighted-----')
# for i in all_w_mean_train:
#     print(i[0])
# print('----------')
# for i in all_w_mean_valid:
#     print(i[0])
    
# print('per case')
errors = pd.read_pickle('ML_models/'+output_main+'/errors_type.pkl')
casetypes = ['DUCT','BUMP','PHLL','CNDV'] #must match load order for numbering
for c,ctype in enumerate(casetypes):
    all_mean_test = np.zeros((len(features)*len(norms),1))
    all_mean_valid = np.zeros((len(features)*len(norms),1))
    all_mean_train = np.zeros((len(features)*len(norms),1))
    all_w_mean_test = np.zeros((len(features)*len(norms),1))
    all_w_mean_valid = np.zeros((len(features)*len(norms),1))
    all_w_mean_train = np.zeros((len(features)*len(norms),1))


    i = 0
    for f_i,f in enumerate(features):
        for n_i,n in enumerate(norms):
            e_temp = errors[errors.columns[(f_i*len(norms)+n_i)*len(casetypes)+c]]
            all_mean_test[i] = np.mean(e_temp[0:2])
            all_mean_valid[i] = np.mean(e_temp[2:4])
            all_mean_train[i] = np.mean(e_temp[4:6])
            all_w_mean_test[i] = np.mean(e_temp[18:20])
            all_w_mean_valid[i] = np.mean(e_temp[20:22])
            all_w_mean_train[i] = np.mean(e_temp[22:24])
            i += 1
    # print(ctype)
    # for i in all_mean_train:
    #     print(i[0])
    # print('----------')
    # for i in all_mean_valid:
    #     print(i[0])
    # print('-------weighted-----')
    # for i in all_w_mean_train:
    #     print(i[0])
    # print('----------')
    # for i in all_w_mean_valid:
    #     print(i[0])
    # for i in range(len(all_w_mean_train)):
    #     print(all_w_mean_train[i,0],all_w_mean_valid[i,0])
        # if i % 2 != 0:

        #     print(all_w_mean_valid[i,0])

    for i in range(len(all_w_mean_train)):
        if i % 2 !=0:
            print(all_w_mean_test[i,0])