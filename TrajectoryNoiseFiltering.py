#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 01:35:46 2018

@author: linux1107pc
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN
from WeaponLibrary import LoadSave
from WeaponLibrary import load_background
import gc
np.random.seed(2018)
# right : 2018
sns.set(style="ticks", font_scale=1.5, color_codes=True)
background = load_background()
###############################################################################
# Visualizing distribution of some features before and after feature-based noise filtering.
def visualizing_before_filtering():
    # Loading original data
    FEATURE_PATH = "..//Data//Original//trajDataOriginalFeatures.pkl"
    
    ls = LoadSave(FEATURE_PATH)
    trajDataFeatures = ls.load_data()
    
    ###################################################
    ###################################################
    
    # Basic feature plot
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(trajDataFeatures["pointNums"], bins=50)
    plt.title("Distribution of the number of points".format(trajDataFeatures["pointNums"].quantile(0.1)))
    plt.savefig("..//Plots//NumberOfPointsDistribution.pdf", bbox_inches='tight')
    
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(trajDataFeatures["totalTime"], bins=50)
    plt.title("Distribution of the total time".format(trajDataFeatures["totalTime"].quantile(0.1)))
    plt.savefig("..//Plots//TotalTimeDistribution.pdf", bbox_inches='tight')
    
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(trajDataFeatures["headTailDist"], bins=50)
    plt.title("Distribution of the head-tail-distance".format(trajDataFeatures["headTailDist"].quantile(0.1)))
    plt.savefig("..//Plots//HeadTailDistanceDistribution.pdf", bbox_inches='tight')
    
    basicReport = trajDataFeatures.describe([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.9, 0.999])
    
    noisyFilter = FeatureBasedNoiseFilter(trajDataFeatures=trajDataFeatures)
    noisyFilter.set_feature_filter_param_lower_bound(headTailDist=20, pointNums=20, totalTime=6)
    noisyFilter.set_feature_filter_param_upper_bound(headTailDist=20000, pointNums=20000, totalTime=20000)
    noiseIndex, noiseCond = noisyFilter._feature_noisy_filter()
    trajDataFeatures.drop(noiseIndex, axis=0, inplace=True)
    
    ###################################################
    ###################################################
    
    print("Nums is {}".format(len(trajDataFeatures)))
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(trajDataFeatures["pointNums"], bins=50)
    plt.title("Distribution of the number of points after filtering".format(trajDataFeatures["pointNums"].quantile(0.1)))
    plt.savefig("..//Plots//NumberOfPointsDistributionAfterFiltering.pdf", bbox_inches='tight')
    
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(trajDataFeatures["totalTime"], bins=50)
    plt.title("Distribution of the total time after filtering".format(trajDataFeatures["totalTime"].quantile(0.1)))
    plt.savefig("..//Plots//TotalTimeDistributionAfterFiltering.pdf", bbox_inches='tight')
    
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(trajDataFeatures["headTailDist"], bins=50)
    plt.title("Distribution of the head-tail-distance after filtering".format(trajDataFeatures["headTailDist"].quantile(0.1)))
    plt.savefig("..//Plots//HeadTailDistanceDistributionAfterFiltering.pdf", bbox_inches='tight')
    plt.close("all")
    del trajDataFeatures
    gc.collect()
    
    return basicReport

###############################################################################
class FeatureBasedNoiseFilter():
    def __init__(self, trajDataFeatures=None):
        self._trajDataFeatures = trajDataFeatures
    
    def set_feature_filter_param_lower_bound(self, headTailDist=10, pointNums=10, totalTime=1):
        self._headTailDistLow = headTailDist
        self._pointNumsLow = pointNums
        self._totalTimeLow = totalTime
    
    def set_feature_filter_param_upper_bound(self, headTailDist=10, pointNums=10, totalTime=1):
        self._headTailDistUp = headTailDist
        self._pointNumsUp = pointNums
        self._totalTimeUp = totalTime
        
    def save_data(self, data=None, fileName=None):
        assert fileName, "Invalid file path !"
        self.__save_data(data, fileName)
        
    def load_data(self, fileName=None):
        assert fileName, "Invalid file path !"
        data = self.__load_data(fileName)
        return data
    
    def __save_data(self, data=None, fileName=None):
        print("--------------Start saving--------------")
        print("Saving file to {}".format(fileName))
        ls = LoadSave(fileName)
        ls.save_data(data)
        print("--------------Saving successed !--------------\n")
        
    def __load_data(self, fileName=None):
        print("--------------Start loading--------------")
        print("Load from {}".format(fileName))
        ls = LoadSave(fileName)
        data = ls.load_data()
        print("--------------loading successed !--------------\n")
        return data
        
    def _feature_noisy_filter(self):
        hdDistNoisyCond = (self._trajDataFeatures["headTailDist"]<self._headTailDistLow) | (self._trajDataFeatures["headTailDist"]>self._headTailDistUp)
        lengthNoisyCond = (self._trajDataFeatures["pointNums"]<self._pointNumsLow) | (self._trajDataFeatures["pointNums"]>self._pointNumsUp)
        totalTimeCond = (self._trajDataFeatures["totalTime"]<self._totalTimeLow) | (self._trajDataFeatures["totalTime"]>self._totalTimeUp)
        noiseCond = hdDistNoisyCond | lengthNoisyCond | totalTimeCond
        noiseIndex = self._trajDataFeatures[noiseCond]["trajIndex"].values
        return noiseIndex, noiseCond
    
if __name__ == "__main__":
    
    ls = LoadSave()
    ls._fileName = "..//Data//Original//trajDataOriginal.pkl"
    trajData = ls.load_data()
    ls._fileName = "..//Data//Original//trajDataOriginalFeatures.pkl"
    trajDataFeatures = ls.load_data()
    
#    visualizing_before_filtering()
    nf = FeatureBasedNoiseFilter(trajDataFeatures=trajDataFeatures)
    nf.set_feature_filter_param_lower_bound(headTailDist=20, pointNums=20, totalTime=6)
    nf.set_feature_filter_param_upper_bound(headTailDist=2000, pointNums=20000, totalTime=20000)
    noiseIndex, noiseCond = nf._feature_noisy_filter()
    
    trajDataFeatures = trajDataFeatures[~noiseCond]
    trajDataFeatures.reset_index(drop=True, inplace=True)
    ls._fileName = "..//Data//Original//trajDataFeaturesAfterNoiseFiltering.pkl"
    ls.save_data(trajDataFeatures)
    
    for ind in noiseIndex:
        trajData.pop(ind)
    ls._fileName = "..//Data//Original//trajDataAfterNoiseFiltering.pkl"
    ls.save_data(trajData)