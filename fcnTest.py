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
################################################################################
#ls = LoadSave(None)
#ls._fileName = "..//Data//TrainData//Completed//trajDataCompleted.pkl"
#trajData = ls.load_data()
from sklearn.neighbors import BallTree
@timefn
def compute_dist_min(traj_1, traj_2):
    if len(traj_1) <= len(traj_2):
        tree = BallTree(traj_1, leaf_size=len(traj_1))
        flag = 1
    else:
        tree = BallTree(traj_2, leaf_size=len(traj_2))
        flag = 2
    
    distMin = np.inf
    p1_ind = None
    p2_ind = None
    if flag == 1:
        for ind_2 in range(len(traj_2)):
            coord = traj_2[ind_2, :].reshape(1, 2)
            distTmp, ind_1 = tree.query(coord)
            if distTmp < distMin:
                distMin = distTmp
                p1_ind = ind_1[0][0]
                p2_ind = ind_2
    else:
        for ind_1 in range(len(traj_1)):
            coord = traj_1[ind_1, :].reshape(1, 2)
            distTmp, ind_2 = tree.query(coord)
            if distTmp < distMin:
                distMin = distTmp
                p1_ind = ind_1
                p2_ind = ind_2[0][0]
    return p1_ind, p2_ind, distMin[0][0]

@timefn
def compute_dist_max(traj_1, traj_2):
    if len(traj_1) >= len(traj_2):
        tree = BallTree(traj_1, leaf_size=len(traj_1))
        flag = 1
    else:
        tree = BallTree(traj_2, leaf_size=len(traj_2))
        flag = 2
    
    distMin = np.inf
    p1_ind = None
    p2_ind = None
    if flag == 1:
        for ind_2 in range(len(traj_2)):
            coord = traj_2[ind_2, :].reshape(1, 2)
            distTmp, ind_1 = tree.query(coord)
            if distTmp < distMin:
                distMin = distTmp
                p1_ind = ind_1[0][0]
                p2_ind = ind_2
    else:
        for ind_1 in range(len(traj_1)):
            coord = traj_1[ind_1, :].reshape(1, 2)
            distTmp, ind_2 = tree.query(coord)
            if distTmp < distMin:
                distMin = distTmp
                p1_ind = ind_1
                p2_ind = ind_2[0][0]
    return p1_ind, p2_ind, distMin[0][0]

@timefn
def brute_compute_closest_distance(traj_1, traj_2):
    distMin = np.inf
    traj_1_length = len(traj_1)
    traj_2_length = len(traj_2)
    
    for ind_1 in range(traj_1_length):
        coord_1 = traj_1[ind_1, :]
        for ind_2 in range(traj_2_length):
            coord_2 = traj_2[ind_2, :]
            distTmp = np.sqrt( np.sum( np.square(coord_1 - coord_2) ) )
            if distTmp < distMin:
                distMin = distTmp
                p1_index, p2_index = ind_1, ind_2
    return p1_index, p2_index, distMin

trajIndex = list(trajData.keys())
traj_1 = trajData[29][["X", "Y"]].values
traj_2 = trajData[11][["X", "Y"]].values
p1, p2, dist = compute_dist_min(traj_1, traj_2)
p1, p2, dist = compute_dist_max(traj_1, traj_2)
p1_brute, p2_brute, dist_brute = brute_compute_closest_distance(traj_1, traj_2)

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