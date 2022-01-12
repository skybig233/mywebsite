# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 11:21
# @Author  : Jiangzhesheng
# @File    : dataset_producer.py
# @Software: PyCharm
# @Description: 1、定义数据集类型2、使用sklearn.datasets生成模拟数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import *

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

    # def load_data(self):
    #     self.data=pd.read_csv(self.path)

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
        with open(self.path,mode='r') as f:
            for line in f:
                print(line)

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


def rand_uniform_data(n:int, d:int)->Dataset:
    """
    随机生成数据
    :param n: 个数
    :param d: 维度
    :return:
    """
    new_set=Dataset()
    new_set.data=np.random.rand(n,d)
    return new_set

def rand_moon_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_moons(n_samples=n,noise=0.07,random_state=120)
    new_set.data=x1
    return new_set

def rand_circles_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_circles(n_samples=n,noise=0.07,random_state=16,factor=0.5)
    new_set.data=x1
    return new_set

def rand_blob_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_blobs(n_samples=n,n_features=2,centers=3,random_state=170)
    new_set.data=x1
    return new_set

def rand_bars_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_blobs(n_samples=n,n_features=2,centers=3,random_state=170)
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    x1 = np.dot(x1, transformation)
    new_set.data=x1
    return new_set

# def pic_data(picpath:str)->Dataset:
#     # TODO 图片转换成模拟的二维数据集
#     #  给定任意图片，先转换成灰度图，
#     #  任意像素点的灰度值为选取到该点的概率，
#     #  以该像素点为圆心，一定阈值（很小）为半径画圆，
#     #  圆内随机取点作为数据点，
#     #  这样就得到了图片的模拟数据集，
#     #  等价于图像灰度化+模糊化？
#
#     """
#
#     :param picpath:
#     :return:
#     """
#     return

def main():
    """
    测试代码
    """
    # a=Dataset(path='kmeans/testdata/testSet.txt')
    # a.load_data()
    # a.normalize_data()
    # a.write_data()
    # a.show_data()

    path='kmeans/testdata/circlesSet.txt'
    new_set=rand_circles_data(300)
    new_set.path=path
    new_set.write_data()

if __name__ == '__main__':
    main()