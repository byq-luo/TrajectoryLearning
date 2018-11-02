#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 01:57:32 2018

@author: linux1107pc
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from WeaponLibrary import load_data
from WeaponLibrary import save_data
from WeaponLibrary import load_background
from NoisyFilter import TrajectoryNoisyFilter
import pickle
from scipy.spatial import ConvexHull
import gc
np.random.seed(2018)
# right : 2018
sns.set(style="ticks", font_scale=1.2, color_codes=True)
background = load_background()
###############################################################################
def core_points_discovery():
    pass


class InterestedRegionDiscovery(object):
    def __init__(self, trajDataFeatures=None, save=False):
        self._trajDataFeatures = trajDataFeatures
        self._save = save
    
    def set_dbscan_start_param(self, eps=None, minSamples=None, alpha=0.5):
        self._startEps = eps
        self._startMinSamples = minSamples
        self._startAlpha = alpha
        
    def set_dbscan_end_param(self, eps=None, minSamples=None, alpha=0.8):
        self._endEps = eps
        self._endMinSamples = minSamples
        self._endAlpha = alpha
        
    def __assign_cluster_label(self, data, eps, minSamples):
        dbscan = DBSCAN(eps=eps, min_samples=minSamples)
        labels = dbscan.fit_predict(data)
        return labels
    
    def _start_pts_noisy_filter(self, plotAll=False):
        startPoints = self._trajDataFeatures[["trajIndex", "trajStartX", "trajStartY"]]
        startCoordinates = startPoints[["trajStartX", "trajStartY"]].values
        corePointFlag = []
        
        # Fit Nearest Neighbors of start points dataset.
        neigh = NearestNeighbors(n_neighbors=self._startMinSamples, n_jobs=1)
        neigh.fit(startCoordinates)
        totalNums = len(startCoordinates)
        
        # Confirm core point flag and assign cluster labels for each data point.
        for ind, pts in enumerate(startCoordinates):
            print("Total is {}, now is {}.".format(totalNums, ind+1))
            pts = pts.reshape(1, -1)
            distance, indices = neigh.kneighbors(pts)
            if distance[:, -1] <= self._startEps:
                corePointFlag.append(1)
            else:
                corePointFlag.append(0)
        startPoints["corePointFlag"] = np.array(corePointFlag)
        startPoints["labels"] = self.__assign_cluster_label(startCoordinates, eps=self._startEps, minSamples=self._startMinSamples)
        self._trajDataFeatures["startZone"] = startPoints["labels"].values
        
        # Calculate core density for each cluster.
        self._start = pd.DataFrame(None, columns=["clusterId", "clusterPointNums", "convexHullArea", "density", "signal"])
        uniqueLabels = list(startPoints[startPoints["labels"] != -1]["labels"].unique())
        clusterId = []
        clusterPointNums = []
        clusterCorePointNums= []
        covexHullArea = []
        density = []
        for i in uniqueLabels:
            clusterId.append(i)
            clusterTmp = startPoints[startPoints["labels"]==i]
            clusterPointNums.append(len(clusterTmp))
            clusterCorePointNums.append(len(clusterTmp[clusterTmp["corePointFlag"]==1]))
            clusterTmp = clusterTmp[clusterTmp["corePointFlag"]==1]
            clusterTmp.drop(["corePointFlag", "labels", "trajIndex"], axis=1, inplace=True)
            corePtsTmp = clusterTmp.values
            hull = ConvexHull(corePtsTmp)
            covexHullArea.append(hull.area)
            density.append(clusterCorePointNums[-1]/hull.area)
        self._start = {"clusterId":clusterId, "clusterCorePointNums":clusterCorePointNums,
                "clusterPointNums":clusterPointNums, "convexHullArea":covexHullArea,
                "density":density}
        self._start = pd.DataFrame(self._start)
        self._startAverageDensity = self._startAlpha * self._start["density"].mean()
        self._start["signal"] = self._start["density"].values > self._startAverageDensity
        
        noisyCluster = self._start["clusterId"][self._start["signal"] != True].values
        
        # Step 1: remove all non core points
        self._trajDataFeatures["startBrokenFlag"] = startPoints["corePointFlag"] == False
        # Step 2: remove all points which are noisy points for DBSCAN
        self._trajDataFeatures["startBrokenFlag"] = self._trajDataFeatures["startBrokenFlag"] | (startPoints["labels"] == -1) 
        for ind in noisyCluster:
            self._trajDataFeatures["startBrokenFlag"] = self._trajDataFeatures["startBrokenFlag"] | (startPoints["labels"] == ind)
        
        # Plot and save all results
        if plotAll == True:
            convexHullPts = startPoints[(startPoints["corePointFlag"] == 1) & (startPoints["labels"] != -1)][["trajStartX", "trajStartY"]].values
            uniqueLabels = list(startPoints[startPoints["labels"] != -1]["labels"].unique())
            labels = startPoints[(startPoints["corePointFlag"] == 1) & (startPoints["labels"] != -1)]["labels"].values
            plt.figure()
            plt.imshow(background)
            density = []
            for ind, item in enumerate(uniqueLabels):
                index = labels == item
                tmp = convexHullPts[index]
                hull = ConvexHull(tmp)
                for simplex in  hull.simplices:
                    plt.plot(tmp[simplex, 0], tmp[simplex, 1], 'k-', linewidth=0.7)
                plt.plot(convexHullPts[index, 0], convexHullPts[index, 1], '.', markersize=1.5, color="red")#RGB[ind])
                pos = convexHullPts[index][2] - 40
                d = len(convexHullPts[index, 0])/hull.area
                d = round(d, 3)
                density.append(d)
                #plt.text(pos[0], pos[1], str(ind) + "_" + str(density[ind]), fontsize=10, color='red') #str(density)
        
            cond = [round(sum(density) * self._startAlpha/(ind+1))]
            plt.title("Entry core points clusters")
            plt.savefig("..//Plots//EntryCorePointsClusteringResults.pdf", dpi=700, bbox_inches='tight')
            plt.figure()
            plt.plot(self._start["clusterId"].astype(int).values, self._start["density"].values, 'k-s')
            x = np.linspace(-1, 7, 2000)
            y = np.array(cond * 2000, dtype='float64')
            plt.plot(x, y, 'b--')
            plt.title("Entry points average density")
            plt.legend(["density", "threshold"])
            plt.xlabel("clusterID")
            plt.ylabel("Core area")
            plt.xlim(self._start["clusterId"].min()-0.1, self._start["clusterId"].max()+0.1)
            plt.ylim(self._start["density"].min()-0.5, self._start["density"].max()+0.5)
            plt.savefig("..//Plots//EntryCorePointsDensityPlots.pdf", dpi=700, bbox_inches='tight')
            plt.close("all")
            
            plt.figure()
            plt.imshow(background)
            normalPts = self._trajDataFeatures[self._trajDataFeatures["startBrokenFlag"] == False][["trajStartX", "trajStartY"]].values
            plt.plot(normalPts[:, 0], normalPts[:, 1], '.', markersize=1.5, color="red")
            plt.title("Start normal total is {}".format(len(normalPts)))
            plt.savefig("..//Plots//EntryCorePointsAfterFiltering.pdf", dpi=700, bbox_inches='tight')
        return self._start
    
    def _end_pts_noisy_filter(self, plotAll=False):
        endPoints = self._trajDataFeatures[["trajIndex", "trajEndX", "trajEndY"]]
        endCoordinates = endPoints[["trajEndX", "trajEndY"]].values
        corePointFlag = []
        neigh = NearestNeighbors(n_neighbors=self._endMinSamples, n_jobs=1)
        neigh.fit(endCoordinates)
        totalNums = len(endCoordinates)
        # Confirm core point flag.
        for ind, pts in enumerate(endCoordinates):
            print("Total is {}, now is {}.".format(totalNums, ind+1))
            pts = pts.reshape(1, -1)
            distance, indices = neigh.kneighbors(pts)
            if distance[:, -1] <= self._endEps:
                corePointFlag.append(1)
            else:
                corePointFlag.append(0)
        endPoints["corePointFlag"] = np.array(corePointFlag)
        endPoints["labels"] = self.__assign_cluster_label(endCoordinates, eps=self._endEps, minSamples=self._endMinSamples)
        self._trajDataFeatures["endZone"] = endPoints["labels"].values
        
        # Calculate core point density.
        self._end = pd.DataFrame(None, columns=["clusterId", "clusterPointNums", "convexHullArea", "density", "signal"])
        uniqueLabels = list(endPoints[endPoints["labels"] != -1]["labels"].unique())
        clusterId = []
        clusterPointNums = []
        clusterCorePointNums= []
        covexHullArea = []
        density = []
        for i in uniqueLabels:
            clusterId.append(i)
            clusterTmp = endPoints[endPoints["labels"]==i]
            clusterPointNums.append(len(clusterTmp))
            clusterCorePointNums.append(len(clusterTmp[clusterTmp["corePointFlag"]==1]))
            clusterTmp = clusterTmp[clusterTmp["corePointFlag"]==1]
            clusterTmp.drop(["corePointFlag", "labels", "trajIndex"], axis=1, inplace=True)
            corePtsTmp = clusterTmp.values
            hull = ConvexHull(corePtsTmp)
            covexHullArea.append(hull.area)
            density.append(clusterCorePointNums[-1]/hull.area)
        self._end = {"clusterId":clusterId, "clusterCorePointNums":clusterCorePointNums,
                "clusterPointNums":clusterPointNums, "convexHullArea":covexHullArea,
                "density":density}
        self._end = pd.DataFrame(self._end)
        self._endAverageDensity = self._endAlpha * self._end["density"].mean()
        self._end["signal"] = self._end["density"].values > self._endAverageDensity
        
        noisyCluster = self._end["clusterId"][self._end["signal"] != True].values
        # Step 1: remove all non core points
        self._trajDataFeatures["endBrokenFlag"] = endPoints["corePointFlag"] == False
        self._trajDataFeatures["endBrokenFlag"] = self._trajDataFeatures["endBrokenFlag"] | (endPoints["labels"] == -1) 
        for ind in noisyCluster:
            self._trajDataFeatures["endBrokenFlag"] = self._trajDataFeatures["endBrokenFlag"] | (endPoints["labels"] == ind)
        
        if plotAll == True:
            convexHullPts = endPoints[(endPoints["corePointFlag"] == 1) & (endPoints["labels"] != -1)][["trajEndX", "trajEndY"]].values
            uniqueLabels = list(endPoints[endPoints["labels"] != -1]["labels"].unique())
            labels = endPoints[(endPoints["corePointFlag"] == 1) & (endPoints["labels"] != -1)]["labels"].values
            plt.figure()
            plt.imshow(background)
            density = []
            for ind, item in enumerate(uniqueLabels):
                index = labels == item
                tmp = convexHullPts[index]
                hull = ConvexHull(tmp)
                for simplex in  hull.simplices:
                    plt.plot(tmp[simplex, 0], tmp[simplex, 1], 'k-', linewidth=0.7)
                plt.plot(convexHullPts[index, 0], convexHullPts[index, 1], '.', markersize=1.5, color="red")#RGB[ind])
                pos = convexHullPts[index][2] - 40
                d = len(convexHullPts[index, 0])/hull.area
                d = round(d, 3)
                density.append(d)
                #plt.text(pos[0], pos[1], str(ind) + "_" + str(density[ind]), fontsize=10, color='red') #str(density)
        
            cond = [sum(density) * self._endAlpha/(ind+1)]
            plt.title("Exit core points clusters")
            density = np.sort(np.array(density))
            plt.savefig("..//Plots//ExitCorePointsClusteringResults.pdf", dpi=700, bbox_inches='tight')
            plt.figure()
            plt.plot(self._end["clusterId"].values, self._end["density"].values, 'k-s')
            x = np.linspace(-1, 7, 2000)
            y = np.array(cond * 2000, dtype='float64')
            plt.plot(x, y, 'b--')
            plt.title("Exit clusters density plot")
            plt.legend(["density", "threshold"])
            plt.xlabel("clusterID")
            plt.ylabel("Core area")
            plt.xlim(self._end["clusterId"].min()-0.1, self._end["clusterId"].max()+0.1)
            plt.ylim(self._end["density"].min()-0.5, self._end["density"].max()+0.5)
            plt.savefig("..//Plots//ExitCorePointsDensityPlots.pdf", dpi=700, bbox_inches='tight')
            plt.close("all")
            
            plt.figure()
            plt.imshow(background)
            normalPts = self._trajDataFeatures[self._trajDataFeatures["endBrokenFlag"] == False][["trajEndX", "trajEndY"]].values
            plt.plot(normalPts[:, 0], normalPts[:, 1], '.', markersize=1.5, color="red")
            plt.title("End normal total is {}".format(len(normalPts)))
            plt.savefig("..//Plots//ExitCorePointsAfterFiltering.pdf", dpi=700, bbox_inches='tight')
        return self._end