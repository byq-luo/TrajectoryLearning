#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 08:42:02 2018

@author: linux1107pc
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

import seaborn as sns
import time
import pickle
from WeaponLibrary import load_background
from WeaponLibrary import plot_list_traj, plot_random_n_traj
from WeaponLibrary import timefn
from WeaponLibrary import sampling_compression
from WeaponLibrary import UnionFind
from WeaponLibrary import LoadSave
from numba import jit
import gc
from DistanceMeasurement import longest_sub_sequence
from DistanceMeasurement import dynamic_time_wraping
from sklearn.cluster import SpectralClustering
from scipy import linalg
import itertools
import multiprocessing
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#warnings.filterwarnings("ignore")

np.random.seed(1)
background = load_background()
###############################################################################
# Exclude the stop points in the completed trajectories and save it into cache
@timefn
def extract_non_stop_traj_data():
    # Load the completed trajectory data 
    ls = LoadSave("..//Data//TrainData//Completed//trajDataCompleted.pkl")
    trajData = ls.load_data()
    
    ls._fileName = "..//Data//TrainData//Completed//trajDataFeaturesCompleted.pkl"
    trajDataFeatures = ls.load_data()
    
    trajDataKeys = list(trajData.keys())
    trajDataNew = {}
    for ind, item in enumerate(trajDataKeys):
        trajGroupby = trajData[item].groupby(["stopIndex"])
        tmp = trajGroupby.mean().reset_index().rename(columns={"index":"stopIndex"})
        tmp = tmp[tmp["stopIndex"] != -1]
        
        # This step is in order to append the average stop points on the new trajectory.
        trajDataNew[item] = trajData[item][trajData[item]["stopIndex"] == -1].reset_index(drop=True)
        trajDataNew[item] = pd.concat([trajDataNew[item], tmp], ignore_index=True)
        trajDataNew[item].sort_values(by="TIME", inplace=True)
        print("Now is {}, total is {}.".format(ind+1, len(trajData)))
    # Save the completed trajectory data without stop points to the cache
    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataWithoutStop.pkl"
    ls.save_data(trajDataNew)
    
    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataWithoutStopFeatures.pkl"
    ls.save_data(trajDataFeatures)
    return trajDataNew

###############################################################################
class WindowCompression():
    def __init__(self, trajData=None):
        self._trajData = trajData
        self._trajDataKeys = list(self._trajData.keys())
    
    def save_data(self, data=None, fileName=None):
        assert data, "Invaild data !"
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
        
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
    
    def set_compression_param(self, samplingNums=None):
        assert samplingNums, "Wrong sampling point nums !"
        self.__set_compression_param(samplingNums=samplingNums)
        
    def __save_data(self, data=None, fileName=None):
        print("--------------Start saving--------------")
        f = open(fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("--------------Saving successed !--------------\n")
        
    def __load_data(self, fileName=None):
        print("--------------Start loading--------------")
        f = open(fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("--------------loading successed !--------------\n")
        return data
    
    def __set_compression_param(self, samplingNums):
        self._samplingNums = samplingNums
    
    def traj_window_compression(self, traj, dist):
        # 接受一条轨迹作为函数的输入，输入类型是numpy coordinates
        # 输出为对应traj长度的索引，索引为轨迹停止点的索引
        trajCompressed = traj
        # 显示采样是否正常，flag=1代表成功；flag=-1代表失败
        flag = 1
        distCumSum = 0
        
        if len(traj) < self._samplingNums:
            print("Too less points ! Want {} but only get {} !".format(self._samplingNums, len(traj)))
            flag = -1
            return trajCompressed, flag
        
        distCumSum = dist.cumsum()
        distCumSum = np.insert(distCumSum, 0, 0)
        totalDistance = distCumSum[-1]
        samplingSequence = np.linspace(0, totalDistance, int(self._samplingNums+1))
        trajCompressedVal = np.zeros((self._samplingNums, 2))
        for i in range(1, len(samplingSequence)):
            cond_1 = (distCumSum >= samplingSequence[i-1])
            cond_2 = (distCumSum < samplingSequence[i])
            cond = cond_1 & cond_2
            if cond.sum() == 0:
                tmp_front = traj[~cond_2].iloc[0].values
                tmp_behind = traj[cond_1].iloc[0].values
                trajCompressedVal[i-1, :] = (tmp_front + tmp_behind)/2
            else:
                trajCompressedVal[i-1, :] = traj[cond].values.mean(axis=0)
        trajCompressed = DataFrame(trajCompressedVal, columns=["X", "Y"])
        return trajCompressed, flag
    
    @timefn
    def compress_trajectory(self):
        self._trajDataCompressed = {}
        for ind in self._trajDataKeys:
            dist = np.sqrt(np.sum(np.square(self._trajData[ind][["X", "Y"]].values[:-1] - self._trajData[ind][["X", "Y"]].values[1:]), axis=1))
            traj = self._trajData[ind][["X", "Y"]]
            trajCompressedTmp, flag = self.traj_window_compression(traj, dist)
            if flag == -1:
                print("Fatal error on {}".format(ind))
                self._trajDataCompressed[ind] = trajCompressedTmp
            else:
                self._trajDataCompressed[ind] = trajCompressedTmp
        return self._trajDataCompressed

###############################################################################
if __name__ == "__main__":
    extract_non_stop_traj_data()
    
    ls = LoadSave("..//Data//TrainData//ClusteringCache//trajDataWithoutStop.pkl")
    trajData = ls.load_data()
    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataWithoutStopFeatures.pkl"
    trajDataFeatures = ls.load_data()
    
    ptsNumList = [55]
    
    for ptsUsedNum in ptsNumList:
        compressor = WindowCompression(trajData)
        compressor.set_compression_param(samplingNums=ptsUsedNum)
        trajDataCompressed = compressor.compress_trajectory()
        
        PATH = "..//Data//TrainData//ClusteringCache//" + "trajDataCompressed_" + str(ptsUsedNum) + ".pkl"
        ls._fileName = PATH
        ls.save_data(trajDataCompressed)