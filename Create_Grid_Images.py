# -*- coding: utf-8 -*-

## ------------------------------------------------------------------------  ##
"""
Objective: determine grid sizes 

Date: 10/19

To Do:
    
    4. clean up the main code
    5. get the features to compute correctly 
"""
## ------------------------------------------------------------------------  ##
" Load Packages "

#general
import numpy as np
import pandas as pd

from utilities.load_dataset import * #load_dataset
from utilities.convert_BUMP import *
from utilities.convert_CNDV import *
from utilities.convert_DUCT import *
from utilities.convert_PHIL import *

import matplotlib.pyplot as plt

#grid stuff
import pyvista as pv
## ------------------------------------------------------------------------  ##
RANS_path='\data\RANSv2'
filename = "BUMP_h31_"
filetype = ".dat"
RANS = load_dataset(RANS_path,filename, filetype, \
                    LES_true=False, LES = False, RANS = True)

x = convert2d_BUMP(RANS,'Cx')
y = convert2d_BUMP(RANS,'Cy')
plt.figure()
plt.pcolor(x,y,x*0,edgecolors='k',cmap='binary')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('bump.png')
print(len(x.reshape(-1,)))
## ------------------------------------------------------------------------  ##
RANS_path='\data\RANS'
filename = "DUCT_1100_"
filetype = ".npy"
RANS = load_dataset(RANS_path,filename, filetype, \
                    LES_true=False, LES = False, RANS = True)

x = convert2d_DUCT(RANS,'Cy')
y = convert2d_DUCT(RANS,'Cz')

plt.figure()
plt.pcolor(x,y,x*0,edgecolors='k',cmap='binary')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('duct.png')
print(len(x.reshape(-1,)))
## ------------------------------------------------------------------------  ##
RANS_path='\data\RANSv2'
filename = "CNDV_12600_"
filetype = ".dat"
RANS = load_dataset(RANS_path,filename, filetype, \
                    LES_true=False, LES = False, RANS = True)

x = convert2d_CNDV(RANS,'Cx')
y = convert2d_CNDV(RANS,'Cy')

plt.figure()
plt.pcolor(x,y,x*0,edgecolors='k',cmap='binary')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('cndv.png')
print(len(x.reshape(-1,)))
## ------------------------------------------------------------------------  ##
RANS_path='\data\RANSv2'
filename = "PHLL_case_1p0_"
filetype = ".dat"
RANS = load_dataset(RANS_path,filename, filetype, \
                    LES_true=False, LES = False, RANS = True)

x = convert2d_PHIL(RANS,'Cx')
y = convert2d_PHIL(RANS,'Cy')

plt.figure()
plt.pcolor(x,y,x*0,edgecolors='k',cmap='binary')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('phll.png')
print(len(x.reshape(-1,)))


a = pd.read_pickle('data/RANS_Dict/all_DICT.pkl')

## ------------------------------------------------------------------------  ##
stop

## ------------------------------------------------------------------------  ##
RANS_path='\data\RANSv2'
filename = "BUMP_h31_"
filetype = ".dat"
RANS = load_dataset(RANS_path,filename, filetype, \
                    LES_true=False, LES = False, RANS = True)
x = convert2d_BUMP(RANS,'Cx')
y = convert2d_BUMP(RANS,'Cy')

plt.figure()
plt.pcolor(x,y,x*0,edgecolors='k',cmap='binary')
plt.plot(x[:,250],y[:,250],color='r')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.figure()
plt.pcolor(x,y,x*0,edgecolors='k',cmap='binary')
plt.plot(x[:,250],y[:,250],color='r')
plt.scatter(x[120,250],y[120,250],color='b')
plt.scatter(x[75,250],y[75,250],color='c')
plt.scatter(x[20,250],y[20,250],color='g')
plt.scatter(x[160,250],y[160,250],color='y')
plt.axis('off')
plt.tight_layout()
plt.xlim([x[0,250]*.99,x[0,250]*1.01])

from features.compute_anisotropy_LES import *
from features.compute_eigenvalues_baycord import *
LES_path='\data\LESv2'
LES = load_dataset(LES_path,filename, ".npy", LES_true=False, LES = True, RANS = False)
LES = compute_anisotropy_LES(LES)
LES = compute_eigenvalues_baycord(LES)
x = convert2d_BUMP(LES,'bay_x')
y = convert2d_BUMP(LES,'bay_y')

plt.figure()
plt.axis('square')
plt.plot([0,1],[0,0],'k')
plt.plot([0,1/2],[0,np.sqrt(3)/2],'k')
plt.plot([1,1/2],[0,np.sqrt(3)/2],'k')
plt.title('Barycentric Tri.')
plt.xlim([-0.1,1.12])
plt.ylim([-0.06,0.93])
plt.axis('off')
plt.scatter(x[:,250],y[:,250],color='r')
plt.scatter(x[120,250],y[120,250],color='b')
plt.scatter(x[75,250],y[75,250],color='c')
plt.scatter(x[20,250],y[20,250],color='g')
plt.scatter(x[160,250],y[160,250],color='y')

plt.figure()
plt.axis('square')
plt.plot([0,1],[0,0],'k')
plt.plot([0,1/2],[0,np.sqrt(3)/2],'k')
plt.plot([1,1/2],[0,np.sqrt(3)/2],'k')
plt.title('Barycentric Tri.')
plt.xlim([-0.1,1.12])
plt.ylim([-0.06,0.93])
plt.axis('off')
plt.scatter(x[:,250],y[:,250],color='r')
