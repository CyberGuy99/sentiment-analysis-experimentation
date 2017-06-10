# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:12:37 2017

@author: rushilcd
"""
import numpy as np

def create_lookup(unig,big,lbl,lookup):
    if lbl==0:
        vec=[1,0,0,0,0]
    elif lbl==1:
        vec=[0,1,0,0,0]
    elif lbl==2:
        vec=[0,0,1,0,0]
    elif lbl==3:
        vec=[0,0,0,1,0]
    else:
        vec=[0,0,0,0,1]
        
    for itm in unig:
        try:
            if lookup.has_key(itm):
                lookup[itm] += np.array(vec)
            else:	
                lookup[itm] = np.array(vec)
        except Exception:
            continue			
        
    for itm in big:
        try:
            if lookup.has_key(itm):
                lookup[itm] += np.array(vec)
            else:	
                lookup[itm] = np.array(vec)
        except Exception:
            continue	
    return None		