# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 21:39:21 2018

@author: Administrator
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
from WeaponLibrary import plot_list_traj
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
                continue
            else:
                self._trajDataCompressed[ind] = trajCompressedTmp
        return self._trajDataCompressed

###############################################################################
class CreateLcssSimilarMatrix():
    def __init__(self, trajDataCompressed=None, fileName="..//Data//window_lcss", load=False):
        # trajData: dict like, value is pandas dataframe
        # paramList: list shape trajectory
        self._trajDataCompressed = trajDataCompressed
        self._trajDataKeys = list(self._trajDataCompressed.keys())
        self._samplingNums = len(self._trajDataCompressed[self._trajDataKeys[0]])
        trajDataCompressedSlice = [self._trajDataCompressed[index][["X", "Y"]].values for index in list(self._trajDataCompressed.keys())]
        self._paramList = list(itertools.combinations(trajDataCompressedSlice, r=2))
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
        lcss = 1 - longest_sub_sequence(params[0], params[1], minDistCond=self._minDistCond, minPts=self._minPts)/ min(len(params[0]), len(params[1]))
        return lcss
    
    def calculate_similar_matrix(self):
        start = time.time()
        pool = multiprocessing.Pool()
        res = pool.map(self.parallel_calculate_traj_lcss, self._paramList)
        end = time.time()
        print("@lcss matrix parallel computing time is {:5f}".format(end-start))
        self._similarMatrix = np.zeros((len(self._trajDataCompressed), len(self._trajDataCompressed)))
        offset = 0
        for row in range(0, len(self._trajDataCompressed)):
            for column in range(row+1, len(self._trajDataCompressed)):
                self._similarMatrix[row, column] = res[offset]
                self._similarMatrix[column, row] = res[offset]
                offset += 1
        return self._similarMatrix
    
    def get_similar_matrix(self):
        self._fileName = self._fileName + "_" + str(len(self._trajDataCompressed)) +"_" + str(self._minDistCond) + "_" + str(self._minPts) + "_" +  "pts" + str(self._samplingNums) +".pkl"
        if self._load == False:
            similarMatrix = self.calculate_similar_matrix()
            self.save_data(self._similarMatrix, fileName=self._fileName)
        else:
            similarMatrix = self.load_data(self._fileName)
        gc.collect()
        return similarMatrix
    
###############################################################################
class RouteClustering(object):
    def __init__(self, trajDataCompressed=None, trajDataFeatures=None, adjacencyMat=None, clusteringNums=30, randomState=15):
        self._trajDataCompressed = trajDataCompressed
        self._trajDataFeatures = trajDataFeatures
        self._adjacencyMat = adjacencyMat
        self._clusteringNums = clusteringNums
        self._randomState = randomState
        self._trajDataIndex = list(self._trajDataCompressed.keys())
        self._trajPtsNums = len(self._trajDataCompressed[self._trajDataIndex[0]])
        
        # 初始化拉普拉斯矩阵相关的参数，包括度矩阵，邻接矩阵和正规化的拉普拉斯矩阵
        self._degreeScalar = np.sum(self._adjacencyMat, axis=0)
        self._degreeMat = np.zeros((self._adjacencyMat.shape[0], self._adjacencyMat.shape[0]))
        self._laplacianMat = np.zeros((self._adjacencyMat.shape[0], self._adjacencyMat.shape[0]))
        for i in range(self._adjacencyMat.shape[0]):
                self._degreeMat[i, i] = self._degreeScalar[i]
                
        self._laplacianMat = self._degreeMat - self._adjacencyMat
        self._laplacianSymMat = (np.sqrt(self._degreeMat)).dot(self._laplacianMat).dot((np.sqrt(self._degreeMat)))
        self._colorCode = ['b', 'lime', 'brown', 'red', 'deeppink', 'darkcyan', 'orange']
        
        # 标志位检测
        self._mergedFlag = 0 #未合并路径
    
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
                 color=colorCode, marker='o', linestyle='None', markersize=1.5)
        
    def plot_clusters(self):
        # 画出所有聚类的结果
        eachPlotTrkNums = 5
        colorCode = self._colorCode
        
        flag = 0
        plt.figure()
        plt.imshow(background)
        for currentLabel in self._trajDataFeatures["originalLabel"].unique():
            if flag <= (eachPlotTrkNums-1):
                for ind, item in enumerate(self._trajDataIndex):
                    if self._trajDataFeatures["originalLabel"][item] == currentLabel:
                        self.plot_one_trajectory(self._trajDataCompressed[item], colorCode=colorCode[flag])
                flag += 1
            else:
                plt.title("Parts of the clustering results")
                #plt.savefig("..//Plots//Clusters_" + str(currentLabel) +".pdf", dpi=500, bbox_inches='tight')
                plt.figure()
                plt.imshow(background)
                flag = 0
                for ind, item in enumerate(self._trajDataIndex):
                    if self._trajDataFeatures["originalLabel"][item] == currentLabel:
                        self.plot_one_trajectory(self._trajDataCompressed[item], colorCode=colorCode[flag])
                flag += 1
                
    def find_abnormal_cluster(self):
        clusterComplexity = self._trajDataFeatures.groupby(["originalLabel"])
        complexityStd = clusterComplexity["totalComplexity"].std()
        self._results["noisyClusters"] = complexityStd.idxmax()
        self._results["noisyNums"] = len(self._trajDataFeatures[self._trajDataFeatures["originalLabel"] == self._results["noisyClusters"]])
        #self._trajDataFeatures["originalLabel"].replace(complexityStd.idxmax(), -1, axis=1, inplace=True)
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
        #plt.savefig("..//Plots//TrajectoryClusteringComplexityStd.png", dpi=600, bbox_inches='tight')
        #plt.close()
        
    def cluster_results_report(self):
        pass
###############################################################################
class RouteMerging():
    def __init__(self, trajDataWindowCompressed=None, trajDataFeatures=None, mergingCond=40):
        self._trajDataWindowCompressed = trajDataWindowCompressed
        self._trajDataFeatures = trajDataFeatures
        self._mergingCond = mergingCond
        self._trajPtsNums = len(self._trajDataWindowCompressed[list(self._trajDataWindowCompressed.keys())[0]])
        
    def save_data(self, data=None, fileName=None):
        #assert data, "Invaild data !"
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
    def set_merged_lcss_param(self, minDistCond=None, minPts=None):
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
        
    def remove_noisy_traj(self):
        noisyIndex = self._trajDataFeatures[self._trajDataFeatures["originalLabel"] == -1]["trajIndex"].values
        for i in noisyIndex:
            self._trajDataWindowCompressed.pop(i)
        self._trajDataFeatures = self._trajDataFeatures[self._trajDataFeatures["originalLabel"] != -1]
        self._trajDataIndex = list(self._trajDataWindowCompressed.keys())
        self._trajDataFeatures.reset_index(inplace=True, drop=True)
    
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
        indexList = np.sort(self._trajDataFeatures["originalLabel"].unique())
        for currentLabel in indexList:
            self._averageCluster[currentLabel] = 0
            trajNums = 0
            for ind, item in enumerate(self._trajDataIndex):
                if self._trajDataFeatures["originalLabel"][ind] == currentLabel:
                    self._averageCluster[currentLabel] += self._trajDataWindowCompressed[item][["X", "Y"]]
                    trajNums += 1
            self._averageCluster[currentLabel] = self._averageCluster[currentLabel]/trajNums
    
    def plot_one_trajectory(self, traj, colorCode=None):
        plt.plot(traj["X"].values, traj["Y"].values, color = colorCode, marker = 'o', linestyle='-', markersize=1.5)
        
    def plot_merged_routes(self):
        plt.figure()
        plt.imshow(background)
        RGB = (np.random.random((len(self._mergedRoutes["mergedLabel"].unique()), 3)))
        for ind, key in enumerate(self._averageRoute.keys()):
            self.plot_one_trajectory(self._averageRoute[key], colorCode=RGB[ind])
        plt.title("Routes after clusters merged")
        plt.savefig("..//Plots//Routes.pdf", dpi=500, bbox_inches='tight')
    
    def plot_unmerged_routes(self):
        plt.figure()
        plt.imshow(background)
        RGB = (np.random.random((len(self._mergedRoutes["originalLabel"].unique()), 3)))
        for ind, key in enumerate(self._averageCluster.keys()):
            self.plot_one_trajectory(self._averageCluster[key], colorCode=RGB[ind])
        plt.title("Routes before clusters merged")
        plt.savefig("..//Plots//RoutesBeforeMerged.pdf", dpi=500, bbox_inches='tight')
    
    def create_lcss_similar_matrix(self, trajData, minDistCond=50, minPts=10):
        trajInd = list(trajData.keys())
        trajNums = len(trajInd)
        similarMatrix = np.zeros((trajNums, trajNums))
        
        for row in range(trajNums):
            for columns in range(row+1, trajNums):
                traj_A = trajData[trajInd[row]][["X", "Y"]].values
                traj_B = trajData[trajInd[columns]][["X", "Y"]].values
                similarMatrix[row, columns] = longest_sub_sequence(traj_A, traj_B, minDistCond=minDistCond, minPts=minPts)
                similarMatrix[columns, row] = similarMatrix[row, columns]
        return similarMatrix
    
    @timefn
    def route_merging(self):
        self.remove_noisy_traj()
        self.calc_average_cluster()
        self._mergedMatrix = self.create_lcss_similar_matrix(self._averageCluster, minDistCond=50, minPts=10)
        
        clusterNums = len(self._trajDataFeatures["originalLabel"].unique())
        indexList = np.arange(0, clusterNums)
        connectedList = []
        mergeCond = self._mergingCond # 重要参数，合并的相似点个数
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
        self.save_data(self._trajDataWindowCompressed, "..//Data//TrajectoryDataRoutesMerged.pkl")
        self.save_data(self._trajDataFeatures, "..//Data//TrajectoryDataFeaturesRoutesMerged.pkl")
        self.save_data(self._averageRoute, "..//Data//TrajectoryDataAverageRoutes.pkl")
        #return self._trajDataWindowCompressed, self._trajDataFeatures

@timefn
def extract_non_stop_traj_data(trajData):
    trajDataKeys = list(trajData.keys())
    trajDataNew = {}
    for ind, item in enumerate(trajDataKeys):
        trajGroupby = trajData[item].groupby(["stopIndex"])
        tmp = trajGroupby.mean().reset_index().rename(columns={"index":"stopIndex"})
        tmp = tmp[tmp["stopIndex"] != -1]
        trajDataNew[item] = trajData[item][trajData[item]["stopIndex"] == -1].reset_index(drop=True)
        trajDataNew[item] = pd.concat([trajDataNew[item], tmp], ignore_index=True)
        trajDataNew[item].sort_values(by="TIME", inplace=True)
        print("Now is {}, total is {}.".format(ind+1, len(trajData)))
    return trajDataNew

#
#def assign_start_end_cluster_label():
#    trajDataFeatures = load_data("..//Data//TrajectoryDataFeaturesAfterFiltering.pkl")
#    kmeans = KMeans(n_clusters=4)
#    
#    # Entry zone discovery
#    trajDataFeatures["entryZone"] = kmeans.fit_predict(trajDataFeatures[["trajStartX", "trajStartY"]].values)
#    entryScore = []
#    clusterNumbers = list(range(2, 10))
#    
#    for i in clusterNumbers:
#        kmeans = KMeans(n_clusters=i)
#        kmeans.fit(trajDataFeatures[["trajStartX", "trajStartY"]].values)
#        labels = kmeans.labels_
#        entryScore.append(silhouette_score(trajDataFeatures[["trajStartX", "trajStartY"]].values, labels))
#    
#    f, ax = plt.subplots(figsize=(8, 6))
#    plt.plot(clusterNumbers, entryScore, 'k-s')
#    plt.title("Silhouette score of entry points")
#    plt.savefig("..//Plots//SilhouetteScoreEntryPoints.pdf", bbox_inches='tight')
#    plt.close("all")
#    
#    # Exit zone discovery
#    kmeans = KMeans(n_clusters=4)
#    trajDataFeatures["exitZone"] = kmeans.fit_predict(trajDataFeatures[["trajEndX", "trajEndY"]].values)
#    exitScore = []
#    for i in clusterNumbers:
#        kmeans = KMeans(n_clusters=i)
#        kmeans.fit(trajDataFeatures[["trajEndX", "trajEndY"]].values)
#        labels = kmeans.labels_
#        exitScore.append(silhouette_score(trajDataFeatures[["trajEndX", "trajEndY"]].values, labels))
#    
#    f, ax = plt.subplots(figsize=(8, 6))
#    plt.plot(clusterNumbers, exitScore, 'k-s')
#    plt.title("Silhouette score of exit points")
#    plt.savefig("..//Plots//SilhouetteScoreExitPoints.pdf", bbox_inches='tight')
#    plt.grid(True)
#    plt.close("all")
#    save_data(trajDataFeatures, "..//Data//TrajectoryDataFeaturesRoutesMerged.pkl")
#    return trajDataFeatures

###############################################################################
if __name__ == "__main__":
    ls = LoadSave("..//Data//Completed//trajDataCompleted.pkl")
    trajData = ls.load_data("..//Data//TrajectoryDataComplete.pkl")
    
    ls._fileName = "..//Data//Completed//trajDataFeaturesCompleted.pkl"
    trajDataFeatures = ls.load_data("..//Data//TrajectoryDataFeaturesComplete.pkl")
    
    trajUsedNums = 500
    trajData = dict([(index, trajData[index]) for index in list(trajData.keys())[:trajUsedNums]])
    trjDataFeaturesSlice = trajDataFeatures.iloc[:trajUsedNums]
    
    trajData = extract_non_stop_traj_data(trajData)
    ptsUsedNums = 50 #120
    # compression
    compressor = WindowCompression(trajData)
    compressor.set_compression_param(samplingNums=ptsUsedNums)
    trajDataCompressed = compressor.compress_trajectory()
    
    # Calculate similar matrix
    c = CreateLcssSimilarMatrix(trajDataCompressed=trajDataCompressed, load=True)
    c.set_lcss_param(minDistCond=60, minPts=5)
    similarMatrix = c.get_similar_matrix()

    eta = 0.35
    adjacencyMat = np.exp( np.divide(-np.square(similarMatrix), 2*eta**2) )

    # clustering
    sc = RouteClustering(trajDataCompressed=trajDataCompressed, trajDataFeatures=trjDataFeaturesSlice, adjacencyMat=adjacencyMat, clusteringNums=30)
    sc.fit_predict()
    trajDataFeatures = sc._trajDataFeatures

    #sc.plot_clusters()
    # Merging routes
    rc = RouteMerging(trajDataWindowCompressed=trajDataCompressed.copy(), trajDataFeatures=trajDataFeatures, mergingCond=45)
    rc.route_merging()
    rc.plot_merged_routes()
    rc.plot_unmerged_routes()
    trajDataFeatures = rc._trajDataFeatures
    gc.collect()
    res_1 = set(trajDataCompressed.keys())
    res_2 = set(trajDataFeatures["trajIndex"].values)
    print(res_1.difference(res_2))
    results = sc._results