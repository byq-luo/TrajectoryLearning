#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 01:23:32 2018

@author: linux1107pc
"""

import pandas as pd
from pandas import DataFrame
import gc
import numpy as np
import pickle
from WeaponLibrary import timefn
from WeaponLibrary import LoadSave
from sklearn.cluster import DBSCAN
np.random.seed(2018)

###############################################################################
def concat_different_files():
    PATH = ["..//Data//Original//trajAll08001000.pkl",
            "..//Data//Original//trajAll10001300.pkl",
            "..//Data//Original//trajAll17001906.pkl"]
    
    trajData = {}
    trajDataIndex = 0
    ls = LoadSave(None)
    
    # trajDataIndex is the new index of each trajectory.
    for ind, path in enumerate(PATH):
        ls._fileName = path
        trajDataTmp = ls.load_data()
        trajDataTmpIndex = list(trajDataTmp.keys())
        
        for i in trajDataTmpIndex:
            trajData[trajDataIndex] = trajDataTmp[i]
            print("Trajectory data index: {}".format(trajDataIndex))
            trajDataIndex += 1
        gc.collect()
    
    ls._fileName = "..//Data//Original//trajDataAll.pkl"
    ls.save_data(trajData)
    return

def load_traj_data():
    PATH = "..//Data//Original//trajDataOriginal.pkl"
    ls = LoadSave(PATH)
    trajData = ls.load_data()
    return trajData

# Reshape trajAll data file
def reshape_traj_data():
    PATH = "..//Data//Original//trajDataAll.pkl"
    ls = LoadSave(PATH)
    trajData = ls.load_data()
    
    trajIndex = list(trajData.keys())
    trajNewData = {}
    
    print("-----------------------------------")
    print("Processing trajAll begins:")
    for ind, item in enumerate(trajIndex):
        print("Now is {}, total is {}.".format(ind+1, len(trajIndex)))
        traj = trajData[item]
        traj["boundingBoxSize"] = traj["W"] * traj["H"]
        traj.drop(["carNum", "maskTraj", "maskCrossTraj", "W", "H"], axis=1, inplace=True)
        traj.rename(columns={"inCrossTraj": "IN", "unixTime":"globalTime"}, inplace=True)
        traj["TIME"] = traj["globalTime"] - traj["globalTime"].iloc[0]
        traj["TIME"] = traj["TIME"].dt.seconds + traj["TIME"].dt.microseconds/1000000
        trajNewData[ind] = traj
        
    ls._fileName = "..//Data//Original//trajDataOriginal.pkl"
    ls.save_data(trajNewData)
    return
###############################################################################

class TrajectoryStopPoints(object):
    def __init__(self, trajData=None, save=False):
        self._trajData = trajData
        self._trajDataIndex = list(trajData.keys())
        self._save = save
        self._trajStopPts = DataFrame(columns=list(self._trajData[self._trajDataIndex[0]].columns))
        
        # Initializing two more attributes
        self._trajStopPts["trajIndex"] = None
        self._trajStopPts["stopIndex"] = None
        
    def save_data(self, data=None, fileName=None):
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
        
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
    
    def set_dbscan_param(self, eps=None, minSamples=None):
        assert eps and minSamples, "Wrong eps and minSamples parameter !"
        self.__set_dbscan_param(eps=eps, minSamples=minSamples)
        
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
    
    def __set_dbscan_param(self, eps=None, minSamples=None):
        self._eps = eps
        self._minSamples = minSamples
    
    @timefn
    def extract_stop_points(self):
        for ind, item in enumerate(self._trajDataIndex):
            trajTmp = self._trajData[item][["X", "Y"]].values
            db = DBSCAN(eps=self._eps, min_samples=self._minSamples, n_jobs=1)
            
            # Prevent some abnormal trajectories
            if len(trajTmp) <= 15000:
                stopPtsIndex = db.fit_predict(trajTmp)
                stopPtsTmp = self._trajData[item][stopPtsIndex!=-1].copy()
                stopPtsTmp["trajIndex"] = item
                stopPtsTmp["stopIndex"] = stopPtsIndex[stopPtsIndex!=-1]
                self._trajStopPts = pd.concat([self._trajStopPts, stopPtsTmp], ignore_index=True)
                print("Stop points extraction :No {} and total is {}".format(ind+1, len(self._trajData)))
                self._trajData[item]["stopIndex"] = stopPtsIndex
            else:
                self._trajData[item]["stopIndex"] = 0
                print("Stop points extraction :No {} and total is {}".format(ind+1, len(self._trajData)))
        if self._save == True:
            self.save_data(self._trajStopPts, "..//Data//Original//TrajectoryDataStopPts.pkl")

###############################################################################
class TrajectoryFeatureExtract(object):
    # self.extract_features()：实例的抽取特征的方法，抽取一系列的轨迹的特征。
    # self.save_data()：实例的保存数据的方法，将抽取的特征保存到本地，同时计算出来的
    # 轨迹累积距离自动保存到本地。
    # self.traj_complexity_directDist_distArray(traj)：计算轨迹复杂度的方法
    def __init__(self, trajData=None, save=False):
        self._trajData = trajData
        self._save = save
        self._trajIndex = list(self._trajData.keys())
    
    def save_data(self, data=None, fileName=None):
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
        
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
    
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
    
    def traj_moving_distance(self, traj):
        trajMoving = traj[traj["stopIndex"] == -1]
        distArray = np.sqrt(np.sum(np.square(trajMoving[["X", "Y"]].values[:-1] - trajMoving[["X", "Y"]].values[1:]), axis=1))
        distCumSum = distArray.cumsum()
        
        # Prevent divide 0 warnings
        actualDistance = distCumSum[-1] if len(distCumSum) != 0 else np.nan
        # Prevent distCumSum[-1] == 0
        return actualDistance if actualDistance != 0 else np.nan
    
    def traj_moving_complexity(self, traj):
        actualDistance = self.traj_moving_distance(traj)
        headTailDist = np.sqrt(np.sum(np.square(traj[["X", "Y"]].iloc[0, :] - traj[["X", "Y"]].iloc[-1, :])))
        ret = headTailDist/actualDistance
        return ret if ret >= 0 else np.nan
    
    def traj_total_distance(self, traj):
        distArray = np.sqrt(np.sum(np.square(traj[["X", "Y"]].values[:-1] - traj[["X", "Y"]].values[1:]), axis=1))
        distCumSum = distArray.cumsum()
        actualDistance = distCumSum[-1] if len(distCumSum) != 0 else np.nan
        return actualDistance if actualDistance != 0 else np.nan
    
    def traj_total_complexity(self, traj):
        actualDistance = self.traj_total_distance(traj)
        headTailDist = np.sqrt(np.sum(np.square(traj[["X", "Y"]].iloc[0, :] - traj[["X", "Y"]].iloc[-1, :])))
        ret = headTailDist/actualDistance
        return ret if ret >= 0 else np.nan
    
    def traj_head_tail_dist(self, traj):
        headTailDist = np.sqrt(np.sum(np.square(traj[["X", "Y"]].iloc[0, :] - traj[["X", "Y"]].iloc[-1, :])))
        return headTailDist
    
    def traj_pts_nums(self, traj):
        return len(traj)
    
    def traj_moving_time(self, traj):
        # 10 points, 0.36 seconds
        ret = (len(traj[traj["stopIndex"] == -1]) - 1) * 0.04
        return ret if ret > 0 else 0
    
    def traj_total_time(self, traj):
        return traj["TIME"].iloc[-1] - traj["TIME"].iloc[0]
    
    def traj_stop_precent(self, traj):
        return len(traj[traj["stopIndex"] != -1])/len(traj)
    
    def traj_stop_segment_nums(self, traj):
        return len(traj["stopIndex"][traj["stopIndex"] != -1].unique())
    
    def traj_box_size_max(self, traj):
        return traj["boundingBoxSize"].max()
    
    def traj_box_size_min(self, traj):
        return traj["boundingBoxSize"].min()
    
    def traj_start_coord(self, traj):
        return traj["X"][0], traj["Y"][0]
    
    def traj_end_coord(self, traj):
        return traj["X"].iloc[-1], traj["Y"].iloc[-1]
    
    def traj_in_intersection_precent(self, traj):
        return traj["IN"].sum()/len(traj["IN"])
    
    def traj_moving_dist_mean(self, traj):
        trajMoving = traj[traj["stopIndex"] == -1]
        distArray = np.sqrt(np.sum(np.square(trajMoving[["X", "Y"]].values[:-1] - trajMoving[["X", "Y"]].values[1:]), axis=1))
        meanDist = distArray.mean()
        return meanDist
    
    def traj_total_dist_mean(self, traj):
        distArray = np.sqrt(np.sum(np.square(traj[["X", "Y"]].values[:-1] - traj[["X", "Y"]].values[1:]), axis=1))
        meanDist = distArray.mean()
        return meanDist
    
    def traj_enter_intersection_time(self, traj):
        trajTmp = traj[traj["IN"] == 1]
        if len(trajTmp) == 0:
            return np.nan
        else:
            return trajTmp["globalTime"].iloc[0]
    
    def traj_leave_intersection_time(self, traj):
        trajTmp = traj[traj["IN"] == 1]
        if len(trajTmp) == 0:
            return np.nan
        else:
            return trajTmp["globalTime"].iloc[-1]
    
    @timefn
    def extract_features(self):
        trajMovingDist = []
        trajMovingComplexity = []
        trajTotalDist = []
        trajTotalComplexity = []
        trajMovingTime = []
        trajTotalTime = []
        trajStopPrecent = []
        trajStopSegmentNums = []
        trajHeadTailDist = []
        trajPtsNums = []
        trajMovingDistMean = []
        trajTotalDistMean = []
        trajEnterTime = []
        trajLeaveTime = []
        
        trajBoxSizeMax = []
        trajBoxSizeMin = []
        trajStartX = []
        trajStartY = []
        trajEndX = []
        trajEndY = []
        trajInPrecent = []
        
        print("\nStart feature extracting:")
        print("----------------------------------------------")
        for item, ind in enumerate(self._trajIndex):
            print("Total is {}, now is {}.".format(len(self._trajIndex), item))
            traj = self._trajData[ind]
            trajMovingDist.append(self.traj_moving_distance(traj))
            trajMovingComplexity.append(self.traj_moving_complexity(traj))
            trajTotalDist.append(self.traj_total_distance(traj))
            trajTotalComplexity.append(self.traj_total_complexity(traj))
            trajMovingTime.append(self.traj_moving_time(traj))
            trajTotalTime.append(self.traj_total_time(traj))
            trajStopPrecent.append(self.traj_stop_precent(traj))
            trajStopSegmentNums.append(self.traj_stop_segment_nums(traj))
            trajInPrecent.append(self.traj_in_intersection_precent(self._trajData[ind]))
            trajHeadTailDist.append(self.traj_head_tail_dist(traj))
            trajPtsNums.append(self.traj_pts_nums(traj))
            trajEnterTime.append(self.traj_enter_intersection_time(traj))
            trajLeaveTime.append(self.traj_leave_intersection_time(traj))
            
            trajMovingDistMean.append(self.traj_moving_dist_mean(traj))
            trajTotalDistMean.append(self.traj_total_dist_mean(traj))
            
            trajBoxSizeMax.append(self.traj_box_size_max(self._trajData[ind]))
            trajBoxSizeMin.append(self.traj_box_size_min(self._trajData[ind]))
            
            trajStartXTmp, trajStartYTmp = self.traj_start_coord(traj)
            trajEndXTmp, trajEndYTmp = self.traj_end_coord(traj)
            trajStartX.append(trajStartXTmp)
            trajStartY.append(trajStartYTmp)
            trajEndX.append(trajEndXTmp)
            trajEndY.append(trajEndYTmp)
        
        self._trajDataFeatures = {"trajIndex":self._trajIndex,
                                  "movingDist":trajMovingDist,
                                  "movingComplexity":trajMovingComplexity,
                                  "totalComplexity":trajTotalComplexity,
                                  "movingTime":trajMovingTime,
                                  "totalTime":trajTotalTime,
                                  "stopPrecent":trajStopPrecent,
                                  "stopSegmentNums":trajStopSegmentNums,
                                  "inPrecent":trajInPrecent,
                                  "headTailDist":trajHeadTailDist,
                                  "pointNums":trajPtsNums,
                                  "boxSizeMax":trajBoxSizeMax,
                                  "boxSizeMin":trajBoxSizeMin,
                                  "movingDistMean":trajMovingDistMean,
                                  "totalDistMean":trajTotalDistMean,
                                  "enterTime":trajEnterTime,
                                  "leaveTime":trajLeaveTime,
                                  
                                  "totalDist":trajTotalDist,
                                  "startX":trajStartX,
                                  "startY":trajStartY,
                                  "endX":trajEndX,
                                  "endY":trajEndY}
        self._trajDataFeatures = DataFrame(self._trajDataFeatures)
        
        self._trajDataFeatures["leaveTime"] = pd.to_datetime(self._trajDataFeatures["leaveTime"], errors='ignore')
        self._trajDataFeatures["enterTime"] = pd.to_datetime(self._trajDataFeatures["enterTime"], errors='ignore')
        print("----------------------------------------------")
        if self._save == True:
            self.save_data(self._trajDataFeatures, '..//Data//Original//trajDataOriginalFeatures.pkl')
            print("Save features sucessed !")
        else:
            return self._trajDataFeatures
###############################################################################
if __name__ == "__main__":
    trajData = load_traj_data()
    
#    stopExtractor = TrajectoryStopPoints(trajData=trajData, save=True)
#    stopExtractor.set_dbscan_param(eps=5, minSamples=40)
#    stopExtractor.extract_stop_points()
#    trajData = stopExtractor._trajData
#    ls = LoadSave("..//Data//Original//trajDataOriginal.pkl")
#    ls.save_data(trajData)
#    
    featureExtractor = TrajectoryFeatureExtract(trajData=trajData, save=True)
    featureExtractor.extract_features()
    trajDataFeatures = featureExtractor._trajDataFeatures