# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:17:56 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from numba import jit
np.random.seed(2018)
###############################################################################
class CBSMOT(object):
    def __init__(self, traj=None, eps=10, minPoints=5, angleThreshold=90):
        # Noisy Points : -1
        # Undefined Points : -100
        self._traj = traj
        self._T = self._traj["TIME"].values
        self._X = self._traj["X"].values
        self._Y = self._traj["Y"].values
        self._angle = self._traj["angle"]
        
        self._eps = eps
        self._minPoints = minPoints
        self._sampleNums = self._traj.shape[0]
        self._clusterId = np.zeros((self._sampleNums, ))
        self._clusterId[:] = -100
        self._index = np.arange(self._sampleNums)
        
        CBSMOT.eps = 5
        CBSMOT.minTime = 10
        self._distArray = np.sqrt(np.square(self._X[:-1] - self._X[1:]) + np.square(self._Y[:-1] - self._Y[1:]))
        self._distCumSum = self._distArray.cumsum()
        
    def _eps_query(self, pointId):
        pass
    
    def _range_query(self, pointId):
        distArray = np.sqrt(np.square(self._X - self._X[pointId]) +
                            np.square(self._Y - self._Y[pointId]))
        index = self._index[distArray < self._eps]
        return index
    
    @jit
    def _expand_cluster(self, currentClusterId, pointId):
        # 查找邻域内的点，返回的是点对应的index
        # 若是返回的index长度和样本数一样，说明RangeQuery失败，抛出AssertError
        index = self._range_query(pointId)
        
        #assert len(index) == self._sampleNums
        if len(index) < self._minPoints:
            self._clusterId[pointId] = -1
            return False
        else:
            self._clusterId[pointId] = currentClusterId
            
            while len(index) > 0:
                currentPointId = index[0]
                index = index[1:]
                if self._clusterId[currentClusterId] == -1:
                    self._clusterId[currentPointId] == currentClusterId
                if self._clusterId[currentPointId] != -100:
                    continue
                self._clusterId[currentPointId] = currentClusterId
                
                currentNeighborIndex = self._range_query(currentPointId)
                # 首先判断是不是Core point，是的话检视CurrentPoint邻域点的标签
                # 若含有Core point并且没有标签，则将Core point以及邻域内的index
                # 加入self._index队列
                if len(currentNeighborIndex) >= self._minPoints:
                    for i in currentNeighborIndex:
                        index = np.append(index, i)
            return True
    
    @jit
    def fit(self):
        clusterId = 0
        for pointId in self._index:
            if self._clusterId[pointId] == -100:
                if self._expand_cluster(clusterId, pointId):
                    clusterId += 1
            else:
                continue
        return self._clusterId
    
'''
m = np.array([[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10], [1.1, 0.8, 1, 4, 3.9, 4.1, 10]])
m = m.T
eps = 0.5
minPts = 2
m = DataFrame(m , columns=["X", "Y"])
db = CBSMOT(m, eps, minPts)
labels = db.fit()
'''