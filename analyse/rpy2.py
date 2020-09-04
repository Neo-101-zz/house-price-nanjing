# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:34:03 2018

@author: Thomas Anderson
"""
# import rpy2's package module
import rpy2.robjects.packages as rpackages
# R vector of strings
from rpy2.robjects.vectors import StrVector
import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()

def get_r_help(topic):
    help_doc = utils.help(topic)
    print(str(help_doc))

def install_r_package(packnames):
    
    # import R's utility package
    utils = rpackages.importr('utils')
    
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    
    # R package names
    for packagename in packnames:
        if len(packagename) > 0:
            utils.install_packages(StrVector(packagename))
        
def dtw_get_distance(ts_1, ts_2):
    # Set up our R namespaces
    R = rpy2.robjects.r
    DTW = importr('dtw')
    
    # Calculate the alignment vector and corresponding distance
    alignment = R.dtw(ts_1, ts_2, keep=True)
    dist = alignment.rx('distance')[0][0]
    
    return dist