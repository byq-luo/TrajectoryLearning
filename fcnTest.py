# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:38:29 2018

@author: XPS13
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import gc
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

import seaborn as sns
from math import acos, pi
from WeaponLibrary import LoadSave
from WeaponLibrary import timefn
from WeaponLibrary import load_background
from WeaponLibrary import plot_list_traj
np.random.seed(1)
sns.set(style="ticks", font_scale=1.2, palette='dark', color_codes=True)
background = load_background()
from CBSMOT import CBSMOT
###############################################################################
def load_data():
    FEATURE_PATH = "..//Data//Original//TrajectoryDataFeatures.pkl"
    TRAJDATA_PATH = "..//Data//Original//TrajectoryDataOriginal.pkl"
    
    ls = LoadSave(FEATURE_PATH)
    trajDataFeatures = ls.load_data()
    ls._fileName = TRAJDATA_PATH
    trajData = ls.load_data()
    return trajData, trajDataFeatures

def traj_angle_calculation(traj):
    trajLength = len(traj)
    if trajLength <= 5:
        traj["angle"] = 0
        return traj
    
    angle = [0, 0]
    for ind in range(2, trajLength-2):
        ptsBehind = traj[["X", "Y"]].iloc[ind-2].values
        ptsCurrent = traj[["X", "Y"]].iloc[ind].values
        ptsNext = traj[["X", "Y"]].iloc[ind+2].values

        dist_1 = np.sqrt( np.sum( np.square( ptsBehind - ptsCurrent ) ) )
        dist_2 = np.sqrt( np.sum( np.square( ptsCurrent - ptsNext ) ) )
        
        vector_1 = ptsCurrent - ptsBehind
        vector_2 = ptsNext - ptsCurrent
        tmp = np.sum(vector_1 * vector_2)
        
        # Prevent zero division error
        if dist_1 != 0 and dist_2 != 0:
            # Prevent cosAngle value error
            try:
                cosAngle = tmp / (dist_1 * dist_2)
                angle.append(acos(cosAngle) if cosAngle <= 1 else 1)
            except:
                print("Value Error !")
                print(cosAngle)
                print(tmp, dist_1, dist_2)
                return
        else:
            angle.append(0)
    angle.append(angle[-1])
    angle.append(angle[-1])
    traj["angle"] = angle
    traj["angle"] = 180/pi * traj["angle"].values
    return angle
        
if __name__ == "__main__":
    trajData, trajDataFeatures = load_data()
    traj = trajData[4836]
    angle = traj_angle_calculation(traj)
    