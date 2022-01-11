# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 11:21
# @Author  : Jiangzhesheng
# @File    : dataset_producer.py
# @Software: PyCharm
# @Description: 生成模拟数据集
import random as rd
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import pylab
FIG_SIZE=(6,6)
class Dataset:

    def __init__(self,path='',data:np.ndarray=None):
        self.path=path
        self.data=data

        if self.path!='':
            self.load_data()

    @property
    def size(self):
        return self.data.shape[0]
    @property
    def dimension(self):
        return self.data.shape[1]

    def load_data(self):
        self.data=pd.read_csv(self.path)

    # def load_data(self)->np.array:
    #     """
    #     将txt文件中的数据转换成np数组
    #     :param filename:模拟数据文件路径
    #     :return:np数组
    #     """
    #     dataset = []
    #     with open(self.path) as file:
    #         for line in file.readlines():
    #             lineArr = line.strip().split('\t')
    #             m = len(lineArr)
    #             dataset.append(lineArr[0:m])
    #         self.data = np.array(dataset, dtype=np.float64)
    #         # print("该数据集有%d个%d维数据" % (data.shape[0], data.shape[1]))
    #     return self.data

    def write_data(self):
        """
        将numpy_ndarray输出到txt文件
        """
        np.savetxt(self.path,self.data,delimiter='\t')
        return

    def normalize_data(self,dimension=[]):
        """
        归一化，最小变为0，最大变为1
        :param dimension: 选择需要归一化的维度，否则全部归一化
        :return:
        """
        dimension=range(self.dimension) if dimension==[] else dimension
        for i in dimension:
            _range = np.max(self.data[:,i]) - np.min(self.data[:,i])
            self.data[:,i]=(self.data[:,i] - np.min(self.data[:,i]))/_range
        return

    def show_data(self):
        self.show_shape()
        # with open(self.path,mode='r') as f:
        #     for line in f:
        #         print(line)

    def show_shape(self):
        print ("该数据集有%d个%d维数据"%(self.size,self.dimension))

    def draw_data(self):
        """
        用matplotlib绘制数据
        """
        self.show_shape()
        # 可视化一下
        plt.scatter(self.data[:, 0], self.data[:, 1], marker='o', color='r')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.show()


def rand_data(n:int,d:int)->Dataset:
    """
    随机生成数据
    :param n: 个数
    :param d: 维度
    :return:
    """
    new_set=Dataset()
    new_set.data=np.random.rand(n,d)
    return new_set

def pic_data(picpath:str)->Dataset:
    return

def main():
    a=Dataset(path='kmeans/testdata/testSet.txt')
    a.load_data()
    # a.normalize_data()
    # a.write_data()
    a.show_data()
    # rand_data(100,2).write_data()

if __name__ == '__main__':
    main()