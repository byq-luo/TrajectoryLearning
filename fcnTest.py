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
from WeaponLibrary import plot_list_traj, plot_random_n_traj
np.random.seed(1)
sns.set(style="ticks", font_scale=1.2, palette='dark', color_codes=True)
background = load_background()
from CBSMOT import CBSMOT
###############################################################################
ls = LoadSave(None)
ls._fileName = "..//Data//Original//trajDataFeaturesAfterNoiseFiltering.pkl"
trajDataFeatures = ls.load_data()
trajDataFeatures.reset_index(drop=True, inplace=True)

def plot_start_end_points(trajDataFeatures=None):
    plt.figure()
    plt.imshow(background)
    plt.plot(trajDataFeatures["startX"], trajDataFeatures["startY"], color='red', marker='.', linestyle=' ', markersize=1.2)
    
    plt.figure()
    plt.imshow(background)
    plt.plot(trajDataFeatures["endX"], trajDataFeatures["endY"], color='red', marker='.', linestyle=' ', markersize=1.2)

plot_start_end_points(trajDataFeatures)

'''
def traj_angle_calculation(traj):
    trajLength = len(traj)
    
    # Prevetn error input data file
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
    #traj = trajData[4836]
    #angle = traj_angle_calculation(traj)
'''     