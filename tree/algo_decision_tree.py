'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
import collections
from math import log
import operator
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dataset_producer import Dataset
from tree.treePlotter import createPlot

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(labelCounts:Dict)->float:
    """
    计算信息熵
    """
    numEntries=sum([labelCounts[i] for i in labelCounts])
    shannonEnt = 0.0
    #遍历label计算信息熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def chooseBestFeatureToSplit(numpy_data:np.ndarray):

    Entropy=collections.namedtuple('Entropy',['ent_value','ent_dict'])#为了方便展示熵的计算
    show_info:List[Entropy,Dict[int,Dict[str,Entropy]]]=[None,{}]#i[0]总信息熵，i[1]条件信息熵

    # numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    numFeatures= numpy_data.shape[1] - 1
    total_d=collections.Counter(numpy_data[:,-1]) #对最后一列标签列进行计数，计算总信息熵
    baseEntropy = calcShannonEnt(total_d)

    show_info[0]=Entropy(baseEntropy,total_d)

    bestInfoGain = 0.0; bestFeature = -1
    #对第i个特征计算条件信息熵和信息增益
    for i in range(numFeatures):        #iterate over all the features
        featList = numpy_data[:,i]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        show_info[1][i]={}
        #第i个特征的每个可取value，计算条件信息熵
        for value in uniqueVals:
            #挑选其中一个value的所有数据
            subDataSet=numpy_data[featList==value,:]
            condition_d=collections.Counter(subDataSet[:,-1])
            prob = len(subDataSet)/float(len(numpy_data))
            condition_ent=calcShannonEnt(condition_d)

            show_info[1][i][value]=Entropy(condition_ent,condition_d)

            newEntropy += prob * condition_ent
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return show_info,bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet:pd.DataFrame):
    """
    这是一个递归算法
    :param dataSet:
    :return:
    """

    numpy_set=dataSet.to_numpy()
    classList = numpy_set[:,-1]
    if len(collections.Counter(classList))==1:
        return classList[0]#stop splitting when all of the classes are equal，剩余标签是同一类
    # if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet，#只剩下一个数据
    #     return majorityCnt(classList)

    ent_info,bestFeat = chooseBestFeatureToSplit(numpy_set)

    bestFeatLabel = dataSet.columns[bestFeat]
    myTree = {bestFeatLabel:{}}

    #对于最好特征那一列的每个可取value
    for value in set(numpy_set[:,bestFeat]):
        #过滤
        filt_set=numpy_set[numpy_set[:,bestFeat]==value,:]
        #用过滤的np.ndarray重新构建df，特征和原来相同，再把最好特征那一列删去
        new_set=pd.DataFrame(data=filt_set,columns=dataSet.columns)
        # new_set=new_set.drop(bestFeatLabel,axis=1)
        myTree[bestFeatLabel][value] = createTree(new_set)

    return myTree
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
if __name__ == '__main__':
    new_data=Dataset(path='./dataset/weather.CSV',haveHead=True)
    calcShannonEnt(new_data.data.to_numpy())
    my_tree = createTree(new_data.data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    createPlot(my_tree)
    print(my_tree)