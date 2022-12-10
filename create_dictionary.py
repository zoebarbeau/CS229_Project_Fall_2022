# -*- coding: utf-8 -*-

## ------------------------------------------------------------------------  ##
"""
Objective: main function for tubrlence anstripoy

Date: 10/14

To Do:
"""
## ------------------------------------------------------------------------  ##
" Load Packages "

#general
import numpy as np
import pandas as pd
import timeit as time

#load feature related functions
from features.compute_anisotropy_LES import *
from features.compute_anisotropy_RANS import *
from features.compute_eigenvalues_baycord import *
from features.compute_tke_threshold import *

from features.plot_BaycentricTri import *

from features.feature_BUMP import *
from features.feature_CNDV import *
from features.feature_DUCT import *
from features.feature_PHIL import *

# load_dataset functions
from utilities.load_dataset import *
from utilities.time_print import *

time_start = time.default_timer()
time_stop = time_start*1.0
## ------------------------------------------------------------------------  ##
" Intialize "
cases_bump = ['BUMP_h20',
         'BUMP_h26',
         'BUMP_h31',
         'BUMP_h38',
         'BUMP_h42']
cases_cndv = ['CNDV_12600',
         'CNDV_20580']

cases_phll = ['PHLL_case_0p5',
         'PHLL_case_0p8',
         'PHLL_case_1p0',
         'PHLL_case_1p2',
         'PHLL_case_1p5']

cases_duct = ['DUCT_1100',
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
         'DUCT_3500']

cases = np.hstack([cases_bump,cases_cndv,cases_phll,cases_duct])

## ------------------------------------------------------------------------  ##
" General Dictionary"
# contains the reynolds numbers and nu of all cases
all_case = pd.DataFrame()
for case in cases:
    # store Reynolds number then nu
    if case.split('_')[0] == 'DUCT':
        Re = int(case.split('_')[1])
        h = 0.5
        u_b = 0.4820
        nu = u_b*h/Re
        all_case[case] = [Re,nu]
    elif case.split('_')[0] == 'PHLL':
        nu =  5E-6
        Re = 5600 # from paper
        all_case[case] = [Re,nu]
    elif case.split('_')[0] == 'BUMP':
        nu = 2.529268E-5
        U = 16.683 #m/s
        h = int(case.split('_')[1].split('h')[1]) # mm
        Re = U*(h*10**-3)/nu
        all_case[case] = [Re,nu]
    else:
        Re = int(case.split('_')[1])
        if Re == 12600:
            nu = 7.9365E-5
        else:
            nu = 4.8591E-5
        all_case[case] = [Re,nu]

all_case.to_pickle('data/RANS_Dict/all_DICT.pkl')

time_stop =  time_print(time_start,time_stop,'General Dict')
## ------------------------------------------------------------------------  ##
" Load Data "
for case in cases:
    # for cases:  BUMP/CNDV/PHIL -> use RANS_path = '\RANSv2' w/ filetype = ".dat"
    # for cases:  DUCT -> use RANS_path = '\RANS' w/ filetype = ".npy"

    filename = case + '_'
    LES_path='\data\LESv2'
    # LES_true_path='\data\LES_true'
    if case.split('_')[0] == 'DUCT':
        RANS_path='\data\RANS'
        filetype = ".npy"
    else:
        RANS_path='\data\RANSv2'
        filetype = ".dat"

    # LES_true = load_dataset(LES_true_path,filename, ".txt", LES_true=True, LES = False, RANS = False)
    LES = load_dataset(LES_path,filename, ".npy", LES_true=False, LES = True, RANS = False)
    RANS = load_dataset(RANS_path,filename, filetype, LES_true=False, LES = False, RANS = True)
    time_stop =  time_print(time_start,time_stop, case +' Load Data')
## ------------------------------------------------------------------------  ##
    " Compute Features"
    RANS = compute_tke_threshold(RANS,Re)
    if case.split('_')[0] == 'DUCT':
        RANS = feature_DUCT(RANS,1,all_case[case][1])
    elif case.split('_')[0] == 'PHLL':
        RANS = feature_PHIL(RANS,1,all_case[case][1])
    elif case.split('_')[0] == 'BUMP':
        RANS = feature_BUMP(RANS,1,all_case[case][1])
    else:
        RANS = feature_CNDV(RANS,1,all_case[case][1])
    time_stop =  time_print(time_start,time_stop,case +' Compute Features')
## ------------------------------------------------------------------------  ##
    " Compute Ansitropy/Eigenvalues/BaycentricCord"
    #1. compute the ansitropy tensor
    LES = compute_anisotropy_LES(LES)
    RANS = compute_anisotropy_RANS(RANS)

    time_stop =  time_print(time_start,time_stop,case +' Compute Ansitropy')

    #2. compute eigenvalues and xy cord.
    LES = compute_eigenvalues_baycord(LES)
    RANS = compute_eigenvalues_baycord(RANS)

    time_stop =  time_print(time_start,time_stop,case +' Compute EigVal/BayCord')

## ------------------------------------------------------------------------  ##
    " Save Dictionary"
    RANS.to_pickle('data/RANS_Dict/'+filename+'DICT.pkl')
    LES.to_pickle('data/LES_Dict/'+filename+'DICT.pkl')

## ------------------------------------------------------------------------  ##
## ------------------------------------------------------------------------  ##

# plot_BaycentricTri(LES['eigVal_1'],LES['eigVal_2'])
#plot_BaycentricTri(RANS['eigVal_1'],RANS['eigVal_2'])
#plot_BaycentricTri(RANS['bay_x'],RANS['bay_y'])

# convert to 2d , and plot norm of xp,yp
# all values that exceed 1 we blank
# from  utilities.convert_DUCT import *
# x = convert2d_DUCT(RANS,'Cy')
# y = convert2d_DUCT(RANS,'Cz')
# z = convert2d_DUCT(RANS,'k')
# z = convert2d_d_DUCT(np.reshape(np.linalg.norm([RANS['bay_x'],RANS['bay_y']],2,axis=0)<1,(-1,1,1)))
# plt.figure()
# plt.scatter(x,y,c=z)
# plt.title('Duct')
# from  utilities.convert_BUMP import *
# x = convert2d_BUMP(RANS,'Cx')
# y = convert2d_BUMP(RANS,'Cy')
# z = convert2d_BUMP(RANS,'k')
# z1 = convert2d_BUMP(RANS,'bay_x')
# z2 = convert2d_BUMP(RANS,'bay_y')
# z = (z1**2 + z2**2)**0.5 < 1
# plt.figure()
# plt.scatter(x,y,c=z)
# plt.title('Bump')
# from  utilities.convert_PHIL import *
# x = convert2d_PHIL(RANS,'Cx')
# y = convert2d_PHIL(RANS,'Cy')
# z = convert2d_PHIL(RANS,'k')
# z1 = convert2d_PHIL(RANS,'bay_x')
# z2 = convert2d_PHIL(RANS,'bay_y')
# z = (z1**2 + z2**2)**0.5 < 1
# plt.figure()
# plt.scatter(x,y,c=z)
# plt.title('PHLL')

# from  utilities.convert_CNDV import *
# x = convert2d_CNDV(RANS,'Cx')
# y = convert2d_CNDV(RANS,'Cy')
# z = convert2d_CNDV(RANS,'k')
# z1 = convert2d_CNDV(RANS,'bay_x')
# z2 = convert2d_CNDV(RANS,'bay_y')
# z = (z1**2 + z2**2)**0.5 < 1
# plt.figure()
# plt.scatter(x,y,c=z)
# plt.title('CNDV')


## ------------------------------------------------------------------------  ##
## ------------------------------------------------------------------------  ##
## ------------------------------------------------------------------------  ##
# load RANS Sij
# S = np.load('komega_BUMP_h20_S.npy')
# k = np.load('komega_BUMP_h20_k.npy')
# omega = np.load('komega_BUMP_h20_omega.npy')
# S = np.load('komega_CNDV_12600_S.npy')
# k = np.load('komega_CNDV_12600_k.npy')
# omega = np.load('komega_CNDV_12600_omega.npy')
# S = np.load('komega_PHLL_case_0p5_S.npy')
# k = np.load('komega_PHLL_case_0p5_k.npy')
# omega = np.load('komega_PHLL_case_0p5_omega.npy')
# S = np.load('komega_DUCT_1100_S.npy')
# k = np.load('komega_DUCT_1100_k.npy')
# omega = np.load('komega_DUCT_1100_omega.npy')


# nut = k/omega

# Sij_dev = np.zeros((np.shape(nut)[0],3,3))
# for i in range(0,2):
#     for j in range(0,2):
#         if i != j:
#             Sij_dev[:,i,j] = S[:,i,j]

# # get Reynolds stress tensor
# Rij = np.zeros((np.shape(nut)[0],3,3))
# for i in range(0,np.shape(nut)[0]):
#     Rij[i,:,:] = 2*nut[i]*Sij_dev[i,:,:]
# # get Ansitropy tensor
# Aij = np.zeros((np.shape(nut)[0],3,3))
# for i in range(0,np.shape(nut)[0]):
#     Aij[i,:,:] = -0.5*Rij[i,:,:]/k[i]

# eigvals = np.zeros((np.shape(nut)[0],3))

# for pt in range(0,np.shape(nut)[0]):
#     A = np.squeeze(Aij[pt,:,:])
#     eigvals[pt,:] = np.sort(np.linalg.eig(A)[0])[::-1]

#     # sort eigenvalues for plottings
#     #eigvals = eigvals[y[:,y_loc].argsort(),:]

# # compute coordinates
# l1 = eigvals[:,0]
# l2 = eigvals[:,1]
# l3 = -l1 - l2
# xp = l1 -l2 + 3/2*l3 + 1/2
# yp = np.sqrt(3)/2*(3*l3+1)

# plot_BaycentricTri(xp,yp)

# stop
