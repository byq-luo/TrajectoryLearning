# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 01:04:04 2018

@author: XPS13
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
import time
from DistanceMeasurement import dynamic_time_wraping

np.random.seed(1)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
background = load_background()
###############################################################################
def brute_valid(ax):
    start = time.time()
    mi = np.inf
    set_1 = ax[ax[:, 3] == 1]
    set_2 = ax[ax[:, 3] == 2]
    for i in set_1:
        for j in set_2:
            d = np.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)
            if d < mi:
                mi = d
                p1, p2 = i, j
    end = time.time()
    print("@MichaelYin, Brute force took {:5f} seconds.".format(end-start))
    return p1, p2, mi


class AssignBrokenTrajectoryLabels(LoadSave):
    def __init__(self, trajDataBroken=None, trajDataFeaturesBroken=None, averageRoutes=None, fileName=None):
        super(AssignBrokenTrajectoryLabels, self).__init__(fileName)
        self._trajDataBroken = trajDataBroken
        self._trajDataFeaturesBroken = trajDataFeaturesBroken
        self._averageRoutes = averageRoutes
    
    @timefn
    def assign_broken_routes(self):
        self.__assign_broken_routes()
        return self._trajDataFeaturesBroken
    
    @jit
    def __assign_broken_routes(self):
        mergedRoutesLabel = []
        brokenKeys = list(self._trajDataBroken.keys())
        routeKeys = list(self._averageRoutes.keys())
        for indKey, trajKey in enumerate(brokenKeys):
            traj = self._trajDataBroken[trajKey][self._trajDataBroken[trajKey]["stopIndex"] == -1]
            if len(traj) == 0:
                mergedRoutesLabel.append(np.nan)
                continue
            distTmp = []
            for indRoute, pathKey in enumerate(routeKeys):
                dtwDistTmp = dynamic_time_wraping(traj[["X", "Y"]].values, self._averageRoutes[pathKey].values)
                distTmp.append(dtwDistTmp)
            distTmp = np.array(distTmp)
            mergedRoutesLabel.append(routeKeys[distTmp.argmin()])
            print("Now is {}, remaining {}.".format(indKey+1, len(self._trajDataBroken)))
        self._trajDataFeaturesBroken["mergedLabel"] = mergedRoutesLabel
    
    def validate(self, index=None):
        # index : list like
        plt.figure()
        plt.imshow(background)
        for ind, item in enumerate(index):
            plt.plot(self._trajDataBroken[item]["X"], self._trajDataBroken[item]["Y"], color='red',  marker=" ",  linestyle='-', markersize=1.5, linewidth=1.4)
            record = self._trajDataFeaturesBroken[self._trajDataFeaturesBroken["trajIndex"] == item]
            
            featureLabel = int(record["mergedLabel"].values)
            plt.plot(self._averageRoutes[featureLabel]["X"], self._averageRoutes[featureLabel]["Y"], color='green', marker=" ",  linestyle='-', markersize=1.5, linewidth=1.4)
        l = plt.legend(["BrokenTrajectory", "Routes"], loc=2, fontsize=5)
        for text in l.get_texts():
            text.set_color("w")
        #plt.axis('off')
        plt.savefig("..//Plots//DTWmergeingResults.pdf", dpi=500, bbox_inches='tight')
        plt.close("all")
        
# Assign broken trajectory routes
if __name__ == "__main__":
    ls = LoadSave(None)
    
    ls._fileName = "..//Data//Broken//trajDataBroken.pkl"
    trajDataBroken = ls.load_data()
    
    ls._fileName = "..//Data//Broken//trajDataFeaturesBroken.pkl"
    trajDataFeaturesBroken = ls.load_data()
    
    ls._fileName = "..//Data//tmpData//trajData.pkl"
    trajDataComplete = ls.load_data()
    
    ls._fileName = "..//Data//tmpData//trajDataFeatures.pkl"
    trajDataFeaturesComplete = ls.load_data()
    
    ls._fileName = "..//Data//tmpData//trajDataAverageRoutes.pkl"
    averageRoutes = ls.load_data()
    
    trajUsedNums = 350
    trajDataBroken = dict([(index, trajDataBroken[index]) for index in list(trajDataBroken.keys())[:trajUsedNums]])
    trajDataFeaturesBroken = trajDataFeaturesBroken.iloc[:trajUsedNums]
    
    clf = AssignBrokenTrajectoryLabels(trajDataBroken=trajDataBroken, trajDataFeaturesBroken=trajDataFeaturesBroken,
                                       averageRoutes=averageRoutes, fileName="..//Data//TrajectoryDataFeaturesBroken_tmp.pkl")
    features = clf.assign_broken_routes()
    trajDataBrokenKeys = list(trajDataBroken.keys())
    clf.validate(index=trajDataBrokenKeys[200:205])

###############################################################################
class ConflictRoutes(LoadSave):
    def __init__(self, averageRoutes=None, trajDataFeaturesComplete=None, fileName=None):
        super(ConflictRoutes, self).__init__(fileName)
        self._averageRoutes = averageRoutes
        self._trajDataFeaturesComplete = trajDataFeaturesComplete
        
    def check_conflicts(self):
        self._conflicDict = self.__possible_conflicts()
        return self._conflicDict
    
    def __weather_intersection(self, traj_1, traj_2, minDistCond=None):
        assert minDistCond, "Invalid minium distance cond !"
        traj_1["traj"] = 1
        traj_2["traj"] = 2
        traj_1.reset_index(inplace=True)
        traj_2.reset_index(inplace=True)
        trajTotal = pd.concat([traj_1, traj_2], ignore_index=True, axis=0)
        trajTotal.reset_index(inplace=True)
        trajTotal.rename(columns={"level_0":"ptsIndex"}, inplace=True)
        
        sortedX = trajTotal[["X", "Y", "index", "traj"]].sort_values(by='X', ascending=True).values
        _, _, d = brute_valid(sortedX.copy())
        if d <= minDistCond:
            return True
        else:
            return False
        
    def __possible_conflicts(self):
        featureGroupby = self._trajDataFeaturesComplete.groupby(["mergedLabel"])
        startEnd = featureGroupby[["startZone", "endZone"]].mean()
        startEnd = startEnd.round()
        startEnd = startEnd.reset_index().rename(columns={"index":"mergedLabel"})
        conflicts = pd.DataFrame(None, columns=["route", "routeConflicts", "conflict"])
        for route in startEnd["mergedLabel"].unique():
            routeStart = startEnd["startZone"][startEnd["mergedLabel"]==route].values[0]
            routeEnd = startEnd["endZone"][startEnd["mergedLabel"]==route].values[0]
            
            # Weather endZone is conflict
            conflictsTmp = startEnd[startEnd["startZone"] != routeStart]["mergedLabel"].values
            #conflictsTmp = startEnd[(startEnd["startZone"] != routeStart) & (startEnd["endZone"] != routeEnd)]["mergedLabel"].values
            tmp = pd.DataFrame(conflictsTmp, columns=["routeConflicts"])
            tmp["route"] = route
            
            traj_1 = self._averageRoutes[route].copy()
            result = []
            for routeConflicts in tmp["routeConflicts"].values:
                traj_2 = self._averageRoutes[routeConflicts].copy()
                result.append(self.__weather_intersection(traj_1.copy(), traj_2.copy(), minDistCond=20))
            tmp["conflict"] = result
            conflicts = pd.concat([conflicts, tmp], ignore_index=True, axis=0)
        conflicts = conflicts[conflicts["conflict"]==True]
        conflicts.drop("conflict", axis=1, inplace=True)
        return conflicts
    
#if __name__ == "__main__":
#    
#    trajDataFeaturesCompleted = load_data("..//Data//tmpData//trajDataFeatures.pkl")
#    averageRoutes = load_data("..//Data//tmpData//trajDataAverageRoutes.pkl")
#
#    clf = ConflictRoutes(averageRoutes=averageRoutes,
#                         trajDataFeaturesComplete=trajDataFeaturesCompleted,
#                         fileName='..//Data//ConflictRoutes.pkl')
#    conflictRoutes = clf.check_conflicts()
#    
#    # Check the conflict result
#    res = conflictRoutes.values
#    for i in range(len(res)):
#        plot_list_traj([res[i, 0], res[i, 1]], averageRoutes)
#        plt.title("Route {} and {}".format(conflictRoute[i, 0], conflictRoute[i, 1]))
#        plt.pause(3)
#        plt.close("all")

###############################################################################
def weather_intersection(traj_1, traj_2, minDistCond=None, minTimeCond=None):
    assert minDistCond, "Invalid ptsCond !"
    assert minTimeCond, "Invalid timeCond !"
    traj_1["traj"] = 1
    traj_2["traj"] = 2
    traj_1.reset_index(inplace=True)
    traj_2.reset_index(inplace=True)
    trajTotal = pd.concat([traj_1, traj_2], ignore_index=True)
    trajTotal.reset_index(inplace=True)
    trajTotal.rename(columns={"level_0":"ptsIndex"}, inplace=True)
     
    sortedX = trajTotal[["X", "Y", "index", "traj"]].sort_values(by='X', ascending=True).values
    #sortedY = trajTotal[["X", "Y", "index", "traj"]].sort_values(by='Y', ascending=True).values
    #p1, p2, d = compute_closest_pair(sortedX.copy(), sortedY.copy())
    p1, p2, d = brute_valid(sortedX.copy())
    
    if d == np.inf:
        return ((p1, p2, d), None, False)
    
    if p1[3] == 1:
        trajTime_1 = traj_1[traj_1["index"]==p1[2]]["unixTime"].iloc[0]
        trajTime_2 = traj_2[traj_2["index"]==p2[2]]["unixTime"].iloc[0]
    else:
        trajTime_1 = traj_1[traj_1["index"]==p2[2]]["unixTime"].iloc[0]
        trajTime_2 = traj_2[traj_2["index"]==p1[2]]["unixTime"].iloc[0]
        
    timeDiff = trajTime_1 - trajTime_2
    timeDiff = abs(timeDiff.total_seconds())
    if d <= minDistCond and timeDiff <= minTimeCond:
        return [(p1, p2, d), timeDiff, True, (trajTime_1, trajTime_2)]
    else:
        return [(p1, p2, d), timeDiff, False, (trajTime_1, trajTime_2)]
    
def validate_two_traj(traj_1, traj_2, p_1, p_2):
    plt.figure()
    plt.imshow(background)
    plt.grid(False)
    plt.plot(traj_1["X"], traj_1["Y"], 'go', markersize=2.2)
    plt.plot(traj_2["X"], traj_2["Y"], 'rs', markersize=2.2)
    plt.plot(p_1[0], p_1[1], color='blue', marker='*', linestyle='', markersize=8)
    plt.plot(p_2[0], p_2[1], color='yellow', marker='^', linestyle='', markersize=8)
    plt.legend(["traj_1", "traj_2", "pt_1", "pt_2"])

def load_all_data():
    trajDataBroken = load_data("..//Data//TrajectoryDataBroken.pkl")
    trajDataComplete = load_data("..//Data//TrajectoryDataComplete.pkl")
    trajDataFeaturesBroken = load_data("..//Data//TrajectoryDataFeaturesBroken_tmp.pkl")
    trajDataFeaturesComplete = load_data("..//Data//TrajectoryDataFeaturesRoutesMerged.pkl")
    averageRoutes = load_data("..//Data//TrajectoryDataAverageRoutes.pkl")
    
    trajDataComplete.update(trajDataBroken)
    trajDataFeatures = pd.concat([trajDataFeaturesBroken, trajDataFeaturesComplete], axis=0, ignore_index=True)
    return trajDataComplete, trajDataFeatures, averageRoutes

def valid_conflict_points(conflictEvent):
    plt.figure()
    plt.imshow(background)
    for ind, item in enumerate(conflictEvent):
        plt.plot(item[0][0][0], item[0][0][1], color='red', marker='o', markersize=2.5)
    plt.title("Point nums:{}".format(len(conflictEvent)))
    

@timefn
def search_conflict_event(trajData, trajDataFeatures, conflictRoutes, minDistCond=15, minTimeCond=3.5):
    # Only keep trajectories that belong to the conflict routes
    routeKeepList = list(conflictRoutes["route"].unique())
    features = pd.DataFrame(None, columns=list(trajDataFeatures.columns))
    for ind, routeInd in enumerate(routeKeepList):
        tmp = trajDataFeatures[trajDataFeatures["mergedLabel"]==routeInd]
        features = pd.concat([features, tmp], ignore_index=True, axis=0)
        
    # Drop some useless trajectories
    features = features[features["inPrecent"] > 0.0001]
    features["inNums"] = features["inPrecent"] * features["pointNums"]
    features = features[features["inNums"] > 5]
    
    # Extracted the conflict trajectory data
    trajDataConflict = dict([(int(index), trajData[int(index)][trajData[int(index)]["IN"] == 1].copy()) for index in features["trajIndex"].unique()])
    del trajData
    gc.collect()
    trajData = trajDataConflict
    
    # Searching conflict record
    conflictEvent = []
    features.sort_values(by='trajIndex', inplace=True)
    trajDataKeys = list(features["trajIndex"].values)
    for ind_1, key_1 in enumerate(trajDataKeys):
        print("\nTotal is {}, now is {}, key is {}.".format(len(trajData), ind_1+1, key_1))
        traj_1 = trajData[key_1].copy()
        traj_1.reset_index(inplace=True, drop=True)
        trajFirstRoute = features[features["trajIndex"] == key_1]["mergedLabel"].values[0]

        featuresCompare = features[features["trajIndex"]>key_1]
        featuresCompare = featuresCompare[(traj_1["unixTime"].iloc[-1] >= featuresCompare["enterTime"]) & (traj_1["unixTime"].iloc[0] <= featuresCompare["leaveTime"])]
        print("Nums is {}".format(len(featuresCompare)))
        trajDataCompareKeys = featuresCompare["trajIndex"].unique()
        for ind_2, key_2 in enumerate(trajDataCompareKeys):
            key_2 = int(key_2)
            trajSecondRoute = features[features["trajIndex"] == key_2]["mergedLabel"].values[0]
            possibleConflict = len(conflictRoutes[(conflictRoutes["route"]==trajFirstRoute) & (conflictRoutes["routeConflicts"]==trajSecondRoute)])
            if possibleConflict == 0:
                continue
            
            traj_2 = trajData[key_2].copy()
            traj_2.reset_index(inplace=True, drop=True)
            res = weather_intersection(traj_1.copy(), traj_2.copy(), minDistCond=minDistCond, minTimeCond=minTimeCond)
            res.append((trajFirstRoute, trajSecondRoute))
            if res[2] == True and possibleConflict:
                print(res)
                res.append([key_1, key_2])
                conflictEvent.append(res)
    return conflictEvent

def transfer_PET_record(data=None, path=None):
    assert path, "Invalid path !"
    coord_1_x = []
    coord_1_y = []
    coord_2_x = []
    coord_2_y = []
    date_1 = []
    date_2 = []
    timeDiff = []
    route_1 = []
    route_2 = []
    minDist = []
    trajIndex_1 = []
    trajIndex_2 = []
    distDiff = []
    
    for ind in data:
        coord_1_x.append(ind[0][0][0])
        coord_1_y.append(ind[0][0][1])
        coord_2_x.append(ind[0][1][0])
        coord_2_y.append(ind[0][1][1])
        date_1.append(ind[3][0])
        date_2.append(ind[3][1])
        timeDiff.append(ind[1])
        route_1.append(ind[4][0])
        route_2.append(ind[4][1])
        minDist.append(ind[0][2])
        trajIndex_1.append(ind[5][0])
        trajIndex_2.append(ind[5][1])
        distDiff.append(ind[0][2])
    conflictEvent = {"coord_1_x":coord_1_x,
                     "coord_1_y":coord_1_y,
                     "coord_2_x":coord_2_x,
                     "coord_2_y":coord_2_y,
                     "date_1":date_1,
                     "date_2":date_2,
                     "timeDiff":timeDiff,
                     "route_1":route_1,
                     "route_2":route_2,
                     "minDist":minDist,
                     "trajIndex_1":trajIndex_1,
                     "trajIndex_2":trajIndex_2,
                     "distDiff":distDiff}
    conflictEvent = pd.DataFrame(conflictEvent)
    saver = LoadSave(fileName=path)
    saver.save_data(conflictEvent)
    return conflictEvent
#Valid all possible conflict routes
#trajData, trajDataFeatures, averageRoutes = load_all_data()
#path = "..//Data//PET_table//PET_3t_2d_327n.pkl"
#loader = LoadSave(fileName=path)
#res = loader.load_data()
#clf = ConflictRoutes(averageRoutes=averageRoutes,
#                     trajDataFeaturesComplete=trajDataFeatures[~trajDataFeatures["originalLabel"].isnull()],
#                     fileName='..//Data//ConflictRoutes.pkl')
#conflictRoutes = clf.check_conflicts()
#conflictEvent = search_conflict_event(trajData, trajDataFeatures, conflictRoutes, minDistCond=2, minTimeCond=3)
#pet = transfer_PET_record(conflictEvent, path=filePath)

#Valid all possible conflict routes
#timeCondList = [1, 1.5, 2, 2.5, 3, 3.5, 4]
#distCondList = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#trajData, trajDataFeatures, averageRoutes = load_all_data()
#clf = ConflictRoutes(averageRoutes=averageRoutes,
#                     trajDataFeaturesComplete=trajDataFeatures[~trajDataFeatures["originalLabel"].isnull()],
#                     fileName='..//Data//ConflictRoutes.pkl')
#conflictRoutes = clf.check_conflicts()
#for timeCond in timeCondList:
#    for distCond in distCondList:
#        conflictEvent = search_conflict_event(trajData, trajDataFeatures, conflictRoutes, minDistCond=distCond, minTimeCond=timeCond)
#        conflictNums = len(conflictEvent)
#        filePath = "..//Data//PET_table//PET_" + str(timeCond) + "t_" + str(distCond) + "d_" + str(conflictNums) + "n.pkl"
#        pet = transfer_PET_record(conflictEvent, path=filePath)
