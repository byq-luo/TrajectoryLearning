#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:46:03 2018

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
from sklearn.metrics import silhouette_score
#warnings.filterwarnings("ignore")

np.random.seed(1)
background = load_background()
###############################################################################
class CreateLcssSimilarMatrix():
    def __init__(self, trajData=None, fileName="..//Data//TrainData//ClusteringCache//lcss", load=False):
        # trajData: dict like, value is pandas dataframe
        # paramList: list shape trajectory
        self._trajData = trajData
        self._trajDataKeys = list(self._trajData.keys())

        self._paramList = list(itertools.combinations(self._trajDataKeys, r=2))
        self._load = load
        self._fileName = fileName
        
    def save_data(self, data=None, fileName=None):
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
        
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
    
    def set_lcss_param(self, minDistCond=None, minPts=None):
        assert minDistCond and minPts, "Invalid, minDistCond and  minPts parameter !"
        self.__set_lcss_param(minDistCond, minPts)
        
    def __save_data(self, data=None, fileName=None):
        print("--------------Start saving...--------------")
        f = open(fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("--------------Saving successed !--------------")
        
    def __load_data(self, fileName=None):
        print("--------------Start loading...--------------")
        f = open(fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("--------------loading successed !--------------")
        return data
    
    def __set_lcss_param(self, minDistCond, minPts):
        self._minDistCond = minDistCond
        self._minPts = minPts
    
    def parallel_calculate_traj_lcss(self, params):
        lcss = 1 - longest_sub_sequence(self._trajData[params[0]][["X", "Y"]].values, self._trajData[params[1]][["X", "Y"]].values,
                                        minDistCond=self._minDistCond, minPts=self._minPts)/ min(len(self._trajData[params[0]][["X", "Y"]].values), len(self._trajData[params[1]][["X", "Y"]].values))
        return lcss
    
    def calculate_similar_matrix(self):
        start = time.time()
        print("@Start parallel computing:")
        pool = multiprocessing.Pool()
        res = pool.map(self.parallel_calculate_traj_lcss, self._paramList)
        end = time.time()
        print("@lcss matrix parallel computing time is {:5f}".format(end-start))
        self._similarMatrix = np.zeros((len(self._trajData), len(self._trajData)))
        offset = 0
        for row in range(0, len(self._trajData)):
            for column in range(row+1, len(self._trajData)):
                self._similarMatrix[row, column] = res[offset]
                self._similarMatrix[column, row] = res[offset]
                offset += 1
        return self._similarMatrix
    
    def get_similar_matrix(self):
        self._fileName = self._fileName + "_" + str(len(self._trajData)) +"_" + str(self._minDistCond) + "_" + str(self._minPts) +".pkl"
        if self._load == False:
            similarMatrix = self.calculate_similar_matrix()
            self.save_data(self._similarMatrix, fileName=self._fileName)
        else:
            similarMatrix = self.load_data(self._fileName)
        gc.collect()
        return similarMatrix

###############################################################################
class RouteClustering(object):
    def __init__(self, trajData=None, trajDataFeatures=None, adjacencyMat=None, clusteringNums=30, randomState=15):
        self._trajData = trajData
        self._trajDataFeatures = trajDataFeatures
        self._adjacencyMat = adjacencyMat
        self._clusteringNums = clusteringNums
        self._randomState = randomState
        self._trajDataIndex = list(self._trajData.keys())
        self._trajDataFeatures.index = self._trajDataFeatures["trajIndex"].values
        # 初始化拉普拉斯矩阵相关的参数，包括度矩阵，邻接矩阵和正规化的拉普拉斯矩阵
        self._degreeScalar = np.sum(self._adjacencyMat, axis=0)
        self._degreeMat = np.zeros((self._adjacencyMat.shape[0], self._adjacencyMat.shape[0]))
        self._laplacianMat = np.zeros((self._adjacencyMat.shape[0], self._adjacencyMat.shape[0]))
        for i in range(self._adjacencyMat.shape[0]):
                self._degreeMat[i, i] = self._degreeScalar[i]
                
        self._laplacianMat = self._degreeMat - self._adjacencyMat
        self._laplacianSymMat = (np.sqrt(self._degreeMat)).dot(self._laplacianMat).dot((np.sqrt(self._degreeMat)))
        self._colorCode = ['b', 'lime', 'brown', 'red', 'deeppink', 'darkcyan', 'orange']
        
    @timefn
    def fit_predict(self):
        sc = SpectralClustering(n_clusters=self._clusteringNums, eigen_solver='arpack',
                                affinity='precomputed', n_init=50,
                                random_state=self._randomState, n_jobs=-1)
        
        sc.fit_predict(self._adjacencyMat)
        self._results = {}
        self._results["affinityMatrix"] = sc.affinity_matrix_
        self._trajDataFeatures["originalLabel"] = sc.labels_
        self._results["eigenValues"], self._results["eigenVectors"] = linalg.eigh(self._laplacianSymMat)
        self._results["eachClusterNums"] = np.bincount(self._trajDataFeatures["originalLabel"].values)
        self.find_abnormal_cluster()
        
    def plot_normalized_eign_map(self, eigenPlotNums=50):
        plt.figure()
        plt.plot(self._results["eigenValues"][:eigenPlotNums], 'k-s', markersize=5)
        plt.title("The first {} eigen values of laplacianSym".format(eigenPlotNums))
        plt.xlim(-0.1)
        plt.ylim(-0.1)
        plt.grid(True)
    
    def plot_one_trajectory(self, traj, colorCode=None):
        plt.plot(traj["X"].values, traj["Y"].values,
                 color=colorCode, marker='.', linestyle='None', markersize=1.5)
        
    def plot_clusters(self):
        # 画出所有聚类的结果
        eachPlotTrkNums = 3
        colorCode = self._colorCode
        
        flag = 0
        plt.figure()
        plt.imshow(background)
        clusterLabels = self._trajDataFeatures["originalLabel"].unique()
        plotLabel = 0 # For naming file convient
        for currentLabel in clusterLabels:
            if flag <= (eachPlotTrkNums-1):
                for ind, item in enumerate(self._trajDataIndex):
                    try:
                        if self._trajDataFeatures["originalLabel"][item] == currentLabel:
                            self.plot_one_trajectory(self._trajData[item], colorCode=colorCode[flag])
                    except:
                        print("\n------------------Error report-----------------------")
                        print("Fatal error, item is {}, currentLabel is {}".format(item, currentLabel))
                        print("Trajectory sample features: {}".format(len(self._trajData[item])))
                        continue
                        print("-------------------------------------------------------")
                flag += 1
            else:
                plt.title("Parts of the clustering results")
                plt.savefig("..//Plots//ClusteringResults//Clusters_" + str(plotLabel) +".png", dpi=500, bbox_inches='tight')
                plotLabel += 1
                plt.figure()
                plt.imshow(background)
                flag = 0
                for ind, item in enumerate(self._trajDataIndex):
                    try:
                        if self._trajDataFeatures["originalLabel"][item] == currentLabel:
                            self.plot_one_trajectory(self._trajData[item], colorCode=colorCode[flag])
                    except:
                        print("\n------------------Error report-----------------------")
                        print("Fatal error, item is {}, currentLabel is {}".format(item, currentLabel))
                        print("Trajectory sample features: {}".format(len(self._trajData[item])))
                        continue
                        print("-------------------------------------------------------")
                flag += 1
        plt.close("all")
        
    def find_abnormal_cluster(self):
        clusterComplexity = self._trajDataFeatures.groupby(["originalLabel"])
        complexityStd = clusterComplexity["movingComplexity"].std()
        self._results["noisyClusters"] = complexityStd.idxmax()
        self._results["noisyNums"] = len(self._trajDataFeatures[self._trajDataFeatures["originalLabel"] == self._results["noisyClusters"]])
        #self._trajDataFeatures["originalLabel"].replace(self._results["noisyClusters"], -1, inplace=True)
        uniqueLabel = self._trajDataFeatures[self._trajDataFeatures["originalLabel"] != -1]["originalLabel"].unique()
        uniqueLabel = np.sort(uniqueLabel)
        for ind, item in enumerate(uniqueLabel):
            self._trajDataFeatures["originalLabel"].replace(item, ind, inplace=True)
        plt.figure()
        plt.plot(complexityStd.values, 'k-s', markersize=6)
        plt.title("Trajectory clusters complexity std")
        plt.xlabel("Cluster label")
        plt.xlim(0)
        plt.ylim(0)
        plt.savefig("..//Plots//ClusteringResults//TrajectoryClusteringComplexityStd.pdf", dpi=600, bbox_inches='tight')
        plt.close()
        
###############################################################################
#if __name__ == "__main__":
#    ls = LoadSave("..//Data//TrainData//ClusteringCache//trajDataCompressed_55.pkl")
#    trajData = ls.load_data()
#    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataWithoutStopFeatures.pkl"
#    trajDataFeatures = ls.load_data()
#    
#    plot_list_traj([12824, 13038], trajData)
#    featureIndexSet = set(list(trajDataFeatures["trajIndex"]))
#    trajIndexSet = set(list(trajData.keys()))
#    print(featureIndexSet.difference(trajIndexSet))
#    trajUsedNums = 4000
#    trajData = dict([(index, trajData[index]) for index in list(trajData.keys())[:trajUsedNums]])
#    trjDataFeaturesSlice = trajDataFeatures.iloc[:trajUsedNums]
#    
#    # Calculate similar matrix
#    c = CreateLcssSimilarMatrix(trajData=trajData, load=True)
#    c.set_lcss_param(minDistCond=40, minPts=5)
#    similarMatrix = c.get_similar_matrix()
#    
#    eta = 0.4
#    adjacencyMat = np.exp( np.divide(-np.square(similarMatrix), 2*eta**2) )
#    
#    # clustering
#    sc = RouteClustering(trajData=trajData, trajDataFeatures=trjDataFeaturesSlice.copy(), adjacencyMat=adjacencyMat, clusteringNums=45)
#    sc.fit_predict()
#    trajDataFeaturesNew = sc._trajDataFeatures
#    sc.plot_clusters()
#    clusterCounts = trajDataFeaturesNew["originalLabel"].value_counts()
#    noiseNums = sc._results["noisyNums"]
#    noiseCluster = sc._results["noisyClusters"]
#    gc.collect()
#    
#    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataFeaturesBeforeMerging.pkl"
#    ls.save_data(trajDataFeaturesNew)
#    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataBeforeMerging.pkl"
#    ls.save_data(trajData)

###############################################################################
class RouteMerging():
    def __init__(self, trajDataWindowCompressed=None, trajDataFeatures=None, matchingPointsDist=40):
        self._trajDataWindowCompressed = trajDataWindowCompressed
        self._trajDataFeatures = trajDataFeatures
        self._matchingPointsDist = matchingPointsDist
        self._trajPtsNums = len(self._trajDataWindowCompressed[list(self._trajDataWindowCompressed.keys())[0]])
        
    def save_data(self, data=None, fileName=None):
        #assert data, "Invaild data !"
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
        
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
        
    def __save_data(self, data=None, fileName=None):
        print("--------------Start saving...--------------")
        print("Save data to path {}.".format(fileName))
        f = open(fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("--------------Saving successed !--------------")
        
    def __load_data(self, fileName=None):
        print("--------------Start loading...--------------")
        print("Load from path {}.".format(fileName))
        f = open(fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("--------------loading successed !--------------")
        return data
    
    def remove_noisy_traj(self):
        noisyIndex = self._trajDataFeatures[self._trajDataFeatures["originalLabel"] == -1]["trajIndex"].values
        for i in noisyIndex:
            self._trajDataWindowCompressed.pop(i)
        self._trajDataFeatures = self._trajDataFeatures[self._trajDataFeatures["originalLabel"] != -1]
        self._trajDataIndex = list(self._trajDataWindowCompressed.keys())
        
    def calc_average_route(self):
        self._averageRoute = {}
        uniqueLabel = np.sort(self._mergedRoutes["mergedLabel"].unique())
        for currentLabel in uniqueLabel:
            self._averageRoute[currentLabel] = 0
            trajNums = 0
            indexList = self._mergedRoutes["originalLabel"][self._mergedRoutes["mergedLabel"] == currentLabel].values
            for ind in indexList:
                self._averageRoute[currentLabel] += self._averageCluster[ind][["X", "Y"]]
                trajNums += 1
            self._averageRoute[currentLabel] = self._averageRoute[currentLabel]/trajNums
    
    def calc_average_cluster(self):
        self._averageCluster = {}
        originalLabelList = np.sort(self._trajDataFeatures["originalLabel"].unique())
        
        for currentLabel in originalLabelList:
            self._averageCluster[currentLabel] = 0
            trajNums = 0
            
            for ind, item in enumerate(self._trajDataIndex):
                if self._trajDataFeatures["originalLabel"][item] == currentLabel and len(self._trajDataWindowCompressed[item] == self._trajPtsNums):
                    self._averageCluster[currentLabel] += self._trajDataWindowCompressed[item][["X", "Y"]]
                    trajNums += 1
            self._averageCluster[currentLabel] = self._averageCluster[currentLabel]/trajNums
    
    def plot_one_trajectory(self, traj, colorCode=None):
        plt.plot(traj["X"].values, traj["Y"].values, color = colorCode, marker = '.', linestyle='-', markersize=0.1)
        
    def plot_merged_routes(self):
        plt.figure()
        plt.imshow(background)
        RGB = (np.random.random((len(self._mergedRoutes["mergedLabel"].unique()), 3)))
        for ind, key in enumerate(self._averageRoute.keys()):
            self.plot_one_trajectory(self._averageRoute[key], colorCode=RGB[ind])
        plt.title("Routes after clusters merged")
        plt.savefig("..//Plots//Routes.pdf", dpi=500, bbox_inches='tight')
    
    def plot_unmerged_clusters(self):
        plt.figure()
        plt.imshow(background)
        RGB = (np.random.random((len(self._averageCluster), 3)))
        for ind, key in enumerate(self._averageCluster.keys()):
            self.plot_one_trajectory(self._averageCluster[key], colorCode=RGB[ind])
        plt.title("Routes before clusters merged")
        plt.savefig("..//Plots//RoutesBeforeMerged.pdf", dpi=500, bbox_inches='tight')
    
    def matching_points(self, traj_A, traj_B):
        assert len(traj_A) == len(traj_B), "Unequal length of two sequence !"
        dist = np.sqrt(np.sum( np.square(traj_A - traj_B), axis=1 ))
        matchingPoins = len(dist[dist <= self._matchingPointsDist])
        return matchingPoins
    
    def create_matching_matrix(self, trajData, minDistCond=50, minPts=10):
        trajInd = list(trajData.keys())
        trajNums = len(trajInd)
        similarMatrix = np.zeros((trajNums, trajNums))
        
        for row in range(trajNums):
            for columns in range(row+1, trajNums):
                traj_A = trajData[trajInd[row]][["X", "Y"]].values
                traj_B = trajData[trajInd[columns]][["X", "Y"]].values
                similarMatrix[row, columns] = self.matching_points(traj_A, traj_B)
                similarMatrix[columns, row] = similarMatrix[row, columns]
        return similarMatrix
    
    @timefn
    def route_merging(self):
        self.remove_noisy_traj()
        self.calc_average_cluster()
        self._mergedMatrix = self.create_matching_matrix(self._averageCluster, minDistCond=50, minPts=10)
        
        clusterNums = len(self._trajDataFeatures["originalLabel"].unique())
        indexList = np.arange(0, clusterNums)
        connectedList = []
        mergeCond = 45 # 重要参数，合并的相似点个数
        
        for i in range(clusterNums):
            similarMatSlice = self._mergedMatrix[:, i]
            sameRoute = indexList[similarMatSlice>=mergeCond]
            for j in sameRoute:
                connectedList.append([i, j])
        uf = UnionFind(clusterNums, connectedList)
        self._mergedRoutes = uf.fit()
        
        self._mergedRoutes = DataFrame(self._mergedRoutes.T, columns=["originalLabel", "mergedLabel"])
        self._trajDataFeatures = pd.merge(self._trajDataFeatures, self._mergedRoutes, on='originalLabel', how='left')
        uniqueLabel = self._trajDataFeatures["mergedLabel"].unique()
        uniqueLabel = np.sort(uniqueLabel)
        for ind, item in enumerate(uniqueLabel):
            self._mergedRoutes["mergedLabel"].replace(item, ind, inplace=True)
            self._trajDataFeatures["mergedLabel"].replace(item, ind, inplace=True)
        self.calc_average_route()
        
        self.save_data(self._trajDataFeatures, "..//Data//TrainData//ClusteringCache//trajDataFeatures.pkl")
        self.save_data(self._averageRoute, "..//Data//TrainData//ClusteringCache//trajDataAverageRoutes.pkl")
###############################################################################
        
if __name__ == "__main__":
    ls = LoadSave(None)
    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataFeaturesBeforeMerging.pkl"
    trajDataFeatures = ls.load_data()
    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataBeforeMerging.pkl"
    trajData = ls.load_data()
    
    rm = RouteMerging(trajDataWindowCompressed=trajData.copy(),
                      trajDataFeatures=trajDataFeatures.copy(),
                      matchingPointsDist=40)
    rm.route_merging()
    
    mergedMat = rm._mergedMatrix
    mergedRoutes = rm._mergedRoutes
    rm.plot_merged_routes()
    features = rm._trajDataFeatures
    completed = features["trajIndex"].values
    trajData = dict([(int(index), trajData[index]) for index in completed])
    ls._fileName = "..//Data//TrainData//ClusteringCache//trajData.pkl"
    ls.save_data(trajData)