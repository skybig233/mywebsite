# -*- coding: utf-8 -*-
# @Time    : 2022/2/11 18:17
# @Author  : Jiangzhesheng
# @File    : kmeans_algo.py
# @Software: PyCharm
# @Description:
from typing import Type

import numpy as np
from collections import namedtuple
from dataset_producer import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS


def randCentroids(x: np.array, k: int) -> np.array:
    """
    随机初始化centroids ,随机选择k个样本点作为质心
    :param x:numpy_data
    :param k:随机选取k个
    :return:选取的中心点
    """
    m, n = x.shape
    randIndex = np.random.choice(m, k)  # 从m个数据中选k个randindex
    centroids = x[randIndex]
    return centroids


# 计算距离
def computeDistance(A, B):
    return np.sqrt(np.sum(np.square(A - B)))


def kmeans(data: np.ndarray, centroids: np.ndarray):
    """
    kmeans核心算法
    :param data: 样本数据
    :param centroids:初始化的质心
    :return:每次分配结果和质心的数据
    """
    K: int = centroids.shape[0]  # 聚类个数
    m: int = data.shape[0]  # 样本个数

    # 记录样本分配的质心的下标、平方误差，初始化为全0
    # clstAss第一列是分配的下标、第二列是SE
    clusterAssment = np.zeros((m, 2))
    clusterchanged: bool = True  # 记录分类是否变化

    # 记录每次循环，样本分配的变化、质心的变化,用于可视化
    info_unit = namedtuple('info_unit', ['clstAss', 'cent'])
    info_list = []

    while clusterchanged:
        clusterchanged = False

        # 1.为每个样本点分配质心
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            # 计算第i个样本到每个中心的距离，取距离最小的
            for j in range(K):
                distance = computeDistance(data[i, :], centroids[j, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterchanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 分配完毕，记录分配结果
        info_list.append(info_unit(clstAss=clusterAssment.copy(), cent=centroids.copy()))

        # 2.更新质心的位置
        for i in range(K):
            index = np.where(clusterAssment[:, 0].ravel() == i)
            centroids[i] = np.mean(data[index], axis=0)
        # 更新完毕，记录更新结果
        info_list.append(info_unit(clstAss=clusterAssment.copy(), cent=centroids.copy()))

    return info_list


def main():
    """
    测试代码
    """
    new_dataset = Dataset('testdata/blobSet.txt')
    centroids = randCentroids(x=new_dataset.data, k=3)
    info_list = kmeans(data=new_dataset.data, centroids=centroids)

    n = -1


    for i in range(3):
        clst_idx = np.where(info_list[n].clstAss[:, 0].ravel() == i)
        clst = new_dataset.data[clst_idx]
        clst_se = info_list[n].clstAss[:, 1][clst_idx]
        plt.scatter(clst[:, 0], clst[:, 1], marker='x', c=list(TABLEAU_COLORS)[i])
        plt.scatter(info_list[n].cent[i, 0], info_list[n].cent[i, 1], marker='o', c=list(TABLEAU_COLORS)[i],
                    linewidths=10)

    plt.show()


if __name__ == '__main__':
    main()
