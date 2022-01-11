# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 11:21
# @Author  : Jiangzhesheng
# @File    : dataset_producer.py
# @Software: PyCharm
# @Description: 生成模拟数据集
import random as rd
import numpy as np
from typing import Tuple

class Dataset:

    def __init__(self,path='',data=[]):
        self.path=path
        self.data=data
    @property
    def size(self):
        return self.data.shape[0]
    @property
    def dimension(self):
        return self.data.shape[1]

    def load_data(self)->np.array:
        """
        将txt文件中的数据转换成np数组
        :param filename:模拟数据文件路径
        :return:np数组
        """
        dataset = []
        with open(self.path) as file:
            for line in file.readlines():
                lineArr = line.strip().split('\t')
                m = len(lineArr)
                dataset.append(lineArr[0:m])
            self.data = np.array(dataset, dtype=np.float64)
            # print("该数据集有%d个%d维数据" % (data.shape[0], data.shape[1]))
        return self.data

    def write_data(self,new_path):
        """
        将numpy_ndarray输出到txt文件
        """
        np.savetxt(new_path,self.data,delimiter='\t')
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

def main():
    a=Dataset(path='testdata/testSet.txt')
    a.load_data()
    a.normalize_data()
    a.write_data(new_path='testdata/testSet.txt.norm')

    rand_data(100,2).write_data(new_path='testdata/randset.txt')

if __name__ == '__main__':
    main()