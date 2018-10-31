# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:01:38 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plt
from numba import jit

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
# 留出参数接口
@jit
def longest_sub_sequence(traj_X, traj_Y, minDistCond=50, minPts=5):
    # 暂定输入为np.array
    trajLength_X = traj_X.shape[0]
    trajLength_Y = traj_Y.shape[0]

    L = np.zeros((trajLength_X+1, trajLength_Y+1))
    for i in range(trajLength_X+1):
        for j in range(trajLength_Y+1):
            if i == 0 or j == 0:
                continue
            else:
                pointDist = np.sqrt(np.square(traj_X[i-1, 0] - traj_Y[j-1, 0]) + 
                                np.square(traj_X[i-1, 1] - traj_Y[j-1, 1]))
            if pointDist < minDistCond and np.abs(i - j) < minPts:
                L[i, j] = L[i-1, j-1] + 1
            else:
                L[i, j] = max(L[i-1, j], L[i, j-1])
    lcss = L[trajLength_X, trajLength_Y]
    return lcss

@jit
def euclidean_distance(traj_X, traj_Y):
    pointDist = np.sqrt(np.square(traj_X[:, 0] - traj_Y[:, 0]) + np.square(traj_X[:, 1] - traj_Y[:, 1]))
    dist = np.sum(pointDist)
    return dist

@jit
def dynamic_time_wraping(traj_X, traj_Y):
    trajLength_X = traj_X.shape[0]
    trajLength_Y = traj_Y.shape[0]
    normConst = trajLength_X + trajLength_Y
    assert trajLength_X, "Invalid X length !"
    assert trajLength_Y, "Invalid Y length !"
    
    D = np.zeros((trajLength_X+1, trajLength_Y+1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    Dtmp = D[1:, 1:]
    for i in range(trajLength_X):
        for j in range(trajLength_Y):
            Dtmp[i, j] = np.sqrt(np.square(traj_X[i, 0] - traj_Y[j, 0]) + np.square(traj_X[i, 1] - traj_Y[j, 1]))
    C = Dtmp.copy()

    for i in range(trajLength_X):
        for j in range(trajLength_Y):
            Dtmp[i, j] += min(D[i, j], D[i, j+1], D[i+1, j])

    i, j = trajLength_X-2, trajLength_Y-2
    coord = []
    while i > 0 or j > 0:
        pos = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if pos == 0:
            i -= 1
            j -= 1
        elif pos == 1:
            i -= 1
        else:
            j -= 1
        coord.append([i, j])
        
    return D[-1, -1]/normConst

@jit
def create_lcss_similar_matrix(trajData, minDistCond, minPts, eta):
    trajInd = list(trajData.keys())
    trajNums = len(trajInd)
    similarMatrix = np.zeros((trajNums, trajNums))
    
    for row in range(trajNums):
        for columns in range(row+1, trajNums):
            traj_A = trajData[trajInd[row]][["X", "Y"]].values
            traj_B = trajData[trajInd[columns]][["X", "Y"]].values
            lcssDistTmp = longest_sub_sequence(traj_A, traj_B, minDistCond=minDistCond, minPts=minPts)
            lcssDistTmp = lcssDistTmp / min(traj_A.shape[0], traj_B.shape[0])
            similarMatrix[row, columns] = 1 - lcssDistTmp
            similarMatrix[columns, row] = similarMatrix[row, columns]
            
    similarMatrix = np.exp( np.divide(-np.square(similarMatrix), 2*eta**2) )
    return similarMatrix