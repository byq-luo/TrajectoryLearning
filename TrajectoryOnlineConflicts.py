#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:46:15 2018

@author: linux1107srv
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import NearestNeighbors
from numba import jit
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

import gc
import seaborn as sns
#import warnings
#warnings.filterwarnings("ignore")
from WeaponLibrary import LoadSave
from WeaponLibrary import timefn
from WeaponLibrary import load_background
from WeaponLibrary import plot_list_traj, plot_random_n_traj
from WeaponLibrary import timefn
import itertools
from DistanceMeasurement import dynamic_time_wraping
np.random.seed(1)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
background = load_background()
###############################################################################
class AssignTestTrajectoryLabels(LoadSave):
    def __init__(self, trajData=None, trajDataFeatures=None, averageRoute=None, averageRouteFeatures=None, fileName=None):
        super(AssignTestTrajectoryLabels, self).__init__(fileName)
        self._trajData = trajData
        self._trajDataFeatures = trajDataFeatures
        self._averageRoute = averageRoute
        
        # Contains average startROI, average endROI and mergedLabel
        self._averageRouteFeatures = averageRouteFeatures
        
    @timefn
    def assign_routes(self):
        self.__assign_routes()
        return self._trajDataFeatures
    
    def __assign_routes(self):
        mergedRoutesLabel = []
        startROI = []
        endROI = []
        
        testKeys = list(self._trajData.keys())
        routeKeys = list(self._averageRoute.keys())
        for ind, trajKey in enumerate(testKeys):
            traj = self._trajData[trajKey][self._trajData[trajKey]["stopIndex"] == -1]
            if len(traj) == 0:
                mergedRoutesLabel.append(np.nan)
                continue
            distTmp = []
            for indRoute, pathKey in enumerate(routeKeys):
                dtwDistTmp = dynamic_time_wraping(traj[["X", "Y"]].values, self._averageRoute[pathKey].values)
                distTmp.append(dtwDistTmp)
            distTmp = np.array(distTmp)
            labelTmp = routeKeys[distTmp.argmin()]
            mergedRoutesLabel.append(labelTmp)
            startROI.append(int(self._averageRouteFeatures["startZone"][ self._averageRouteFeatures["mergedLabel"] == labelTmp ].values))
            endROI.append(int(self._averageRouteFeatures["endZone"][ self._averageRouteFeatures["mergedLabel"] == labelTmp ].values))
            
            print("Now is {}, remaining {}.".format(ind+1, len(self._trajData)))
        self._trajDataFeatures["mergedLabel"] = mergedRoutesLabel
        self._trajDataFeatures["startZone"] = startROI
        self._trajDataFeatures["endZone"] = endROI
        
    def validate(self, index=None):
        # index : list like
        plt.figure()
        plt.imshow(background)
        for ind, item in enumerate(index):
            plt.plot(self._trajData[item]["X"], self._trajData[item]["Y"], color='red',  marker=" ",  linestyle='-', markersize=1.5, linewidth=1.4)
            record = self._trajDataFeaturesBroken[self._trajDataFeaturesBroken["trajIndex"] == item]
            
            featureLabel = int(record["mergedLabel"].values)
            plt.plot(self._averageRoute[featureLabel]["X"], self._averageRoute[featureLabel]["Y"], color='green', marker=" ",  linestyle='-', markersize=1.5, linewidth=1.4)
        l = plt.legend(["BrokenTrajectory", "Routes"], loc=2, fontsize=5)
        for text in l.get_texts():
            text.set_color("w")
        #plt.axis('off')
        plt.savefig("..//Plots//DTWmergeingResults.pdf", dpi=500, bbox_inches='tight')
        plt.close("all")

###############################################################################
class ConflictRoutes(LoadSave):
    def __init__(self, averageRoute=None, averageRouteFeatures=None, fileName=None):
        super(ConflictRoutes, self).__init__(fileName)
        self._averageRoute = averageRoute
        self._averageRouteFeatures = averageRouteFeatures
        
    def check_conflicts(self):
        self._conflicDict = self.__possible_conflicts()
        return self._conflicDict

    def brute_force(self, traj_1=None, traj_2=None, distCond=20):
        for coord_1 in traj_1:
            for coord_2 in traj_2:
                distanceTmp = np.sqrt(np.sum( np.square(coord_1 - coord_2) ))
                if distanceTmp <= distCond:
                    return True
        return False
    
    @timefn
    def __possible_conflicts(self):
        routeUnique = self._averageRouteFeatures["mergedLabel"].unique()
        conflictCheckList = list(itertools.combinations(routeUnique, r=2))
        route = []
        routeConflict = []
        conflictType = []
        res = {}
        
        for ind in conflictCheckList:
            routeLabel_1 = ind[0]
            routeLabel_2 = ind[1]
            route_1 = self._averageRoute[routeLabel_1].values
            route_2 = self._averageRoute[routeLabel_2].values
            
            startROI_1 = self._averageRouteFeatures["startZone"][self._averageRouteFeatures["mergedLabel"]==routeLabel_1].values[0]
            startROI_2 = self._averageRouteFeatures["startZone"][self._averageRouteFeatures["mergedLabel"]==routeLabel_2].values[0]
            endROI_1 = self._averageRouteFeatures["endZone"][self._averageRouteFeatures["mergedLabel"]==routeLabel_1].values[0]
            endROI_2 = self._averageRouteFeatures["endZone"][self._averageRouteFeatures["mergedLabel"]==routeLabel_2].values[0]
            
            # Weather conflict
            conflictRes = self.brute_force(route_1, route_2, distCond=20)
            
            if conflictRes == True:
                route.append(routeLabel_1)
                routeConflict.append(routeLabel_2)
                # 1: The following conflict
                # 2: The direct conflict
                # 3: The merging conflict
                
                if startROI_1 == startROI_2:
                    conflictType.append(1)
                elif startROI_1 != startROI_2 and endROI_1 != endROI_2:
                    conflictType.append(2)
                elif startROI_1 != startROI_2 and endROI_1 == endROI_2:
                    conflictType.append(3)
            elif conflictRes == False:
                pass
        res["route"] = route
        res["routeConflict"] = routeConflict
        res["conflictType"] = conflictType
        conflicts = pd.DataFrame(res)
        return conflicts
###############################################################################
from sklearn.neighbors import BallTree
def compute_dist_tree(traj_1, traj_2):
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

def weather_intersection(traj_1, traj_2, minDistCond=None, minTimeCond=None):
    assert minDistCond, "Invalid ptsCond !"
    assert minTimeCond, "Invalid timeCond !"
    p1_ind, p2_ind, d = compute_dist_tree(traj_1[["X", "Y"]].values, traj_2[["X", "Y"]].values)
    
    if d == np.inf:
        return ((p1_ind, p2_ind, d), None, False)
    trajTime_1 = traj_1.iloc[p1_ind]["globalTime"]
    trajTime_2 = traj_2.iloc[p2_ind]["globalTime"]
    timeDiff = trajTime_1 - trajTime_2
    timeDiff = abs(timeDiff.total_seconds())
    
    if d <= minDistCond and timeDiff <= minTimeCond:
        return [traj_1.iloc[p1_ind][["X", "Y"]].values, traj_2.iloc[p2_ind][["X", "Y"]].values, d, timeDiff, True, trajTime_1, trajTime_2]
    else:
        return [traj_1.iloc[p1_ind][["X", "Y"]].values, traj_2.iloc[p2_ind][["X", "Y"]].values, d, timeDiff, False, trajTime_1, trajTime_2]

@timefn
def searching_conflict_events(trajData, trajDataFeatures, conflictRoutes, minDistCond=15, minTimeCond=3.5):
    # Only keep trajectories that belong to the conflict routes
    routeKeepList = list(conflictRoutes["route"].unique())
    uniqueRouteLabel = trajDataFeatures["mergedLabel"].unique()
    for route in uniqueRouteLabel:
        if route not in routeKeepList:
            trajDataFeatures = trajDataFeatures[trajDataFeatures["mergedLabel"] != route]
    features = trajDataFeatures
    
    # Drop some useless trajectories
    features = features[features["inPrecent"] > 0.01]
    features["inNums"] = features["inPrecent"].values * features["pointNums"].values
    features = features[features["inNums"] > 10]
    
    # Extracted the conflict trajectory data in the intersection
    trajDataIn = dict([(int(index), trajData[int(index)][trajData[int(index)]["IN"] == 1].copy()) for index in features["trajIndex"].unique()])
    trajData = trajDataIn
    gc.collect()
    
    # Searching conflict record
    conflictEvent = []
    features.sort_values(by='trajIndex', inplace=True)
    trajDataKeys = list(features["trajIndex"].values)
    
    coord_1_x = []
    coord_1_y = []
    coord_2_x = []
    coord_2_y = []
    date_1 = []
    date_2 = []
    timeDiff = []
    routeLabel_1 = []
    routeLabel_2 = []
    minDist = []
    trajIndex_1 = []
    trajIndex_2 = []
    
    for ind_1, key_1 in enumerate(trajDataKeys):
        print("Total is {}, now is {}, key is {}.".format(len(trajData), ind_1+1, key_1))
        traj_1 = trajData[key_1].copy()
        traj_1.reset_index(inplace=True, drop=True)
        trajFirstRouteLabel = features[features["trajIndex"] == key_1]["mergedLabel"].values[0]
        
        # Time condition
        featuresCompare = features[features["trajIndex"]>key_1]
        featuresCompare = featuresCompare[(traj_1["globalTime"].iloc[-1] >= featuresCompare["enterInterTime"]) & (traj_1["globalTime"].iloc[0] <= featuresCompare["leaveInterTime"])]
        
        # Routes condition
#        routeConflict = conflictRoutes["routeConflict"][conflictRoutes["route"] == trajFirstRouteLabel].values
#        uniqueLabel = featuresCompare["mergedLabel"].unique()
#        for label in uniqueLabel:
#            if label not in routeConflict:
#                featuresCompare = featuresCompare[featuresCompare["mergedLabel"] != uniqueLabel]
#            
        print("Compare nums is {}".format(len(featuresCompare)))
        
        
        trajDataCompareIndex = featuresCompare["trajIndex"].unique()
        for ind_2, key_2 in enumerate(trajDataCompareIndex):
            key_2 = int(key_2)
            trajSecondRouteLabel = features[features["trajIndex"] == key_2]["mergedLabel"].values[0]
            possibleConflict = len(conflictRoutes[(conflictRoutes["route"]==trajFirstRouteLabel) & (conflictRoutes["routeConflict"]==trajSecondRouteLabel)])
#            if possibleConflict == 0:
#                continue
            possibleConflict = 1
            traj_2 = trajData[key_2].copy()
            traj_2.reset_index(inplace=True, drop=True)
            res = weather_intersection(traj_1.copy(), traj_2.copy(), minDistCond=minDistCond, minTimeCond=minTimeCond)
            res.append((trajFirstRouteLabel, trajSecondRouteLabel))
            if res[4] == True and possibleConflict:
                coord_1_x.append(res[0][0])
                coord_1_y.append(res[0][1])
                coord_2_x.append(res[1][0])
                coord_2_y.append(res[1][1])
                minDist.append(res[2])
                timeDiff.append(res[3])
                date_1.append(res[5])
                date_2.append(res[6])
                routeLabel_1.append(trajFirstRouteLabel)
                routeLabel_2.append(trajSecondRouteLabel)
                trajIndex_1.append(key_1)
                trajIndex_2.append(key_2)
                
        conflictEvent = {"coord_1_x":coord_1_x,
                         "coord_1_y":coord_1_y,
                         "coord_2_x":coord_2_x,
                         "coord_2_y":coord_2_y,
                         "date_1":date_1,
                         "date_2":date_2,
                         "timeDiff":timeDiff,
                         "routeLabel_1":routeLabel_1,
                         "routeLabel_2":routeLabel_2,
                         "minDist":minDist,
                         "trajIndex_1":trajIndex_1,
                         "trajIndex_2":trajIndex_2}
        conflictEvent = pd.DataFrame(conflictEvent)
    return conflictEvent
###############################################################################
if __name__ == "__main__":
#    ls = LoadSave(None)
#    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataFeatures.pkl"
#    trajDataFeatures = ls.load_data()
#    ls._fileName = "..//Data//TrainData//ClusteringCache//trajDataAverageRoutes.pkl"
#    averageRoute = ls.load_data()
#    featureGroupby = trajDataFeatures.groupby(trajDataFeatures["mergedLabel"])
#    averageRouteFeatures = np.round(featureGroupby[["startZone", "endZone", "mergedLabel"]].mean())
#    
#    clf = ConflictRoutes(averageRoute=averageRoute, averageRouteFeatures=averageRouteFeatures)
#    conflicts = clf.check_conflicts()
#    
#    ls._fileName = "..//Data//TestData//AllData//trajDataTest.pkl"
#    trajDataTest = ls.load_data()
#    ls._fileName = "..//Data//TestData//AllData//trajDataTestFeaturesAssigned.pkl"
#    trajDataTestFeatures = ls.load_data()
    
    res = searching_conflict_events(trajDataTest, trajDataTestFeatures.copy(), conflicts, minDistCond=4, minTimeCond=3)
    
    
    