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
from collections import Counter
FIG_SIZE=(6,6)
class Dataset:

    def __init__(self, path='', data:np.ndarray=None, isLabeled=False, haveHead=False):
        self.path=path
        self.data=data
        #load则label是列表，随机生成则label是np.ndarray
        self.label=[]

        self.head=[]
        if self.path!='':
            self.load_data(isLabeled=isLabeled,haveHead=haveHead)

    @property
    def size(self):
        return self.data.shape[0]
    @property
    def dimension(self):
        return self.data.shape[1]
    @property
    def label_count(self):
        return Counter(self.label)

    def load_data(self, isLabeled=False,haveHead=False)->np.array:
        """
        将txt文件中的数据转换成np数组，存入data属性中，并返回data属性值
        :param filename:模拟数据文件路径
        :return:np数组
        """

        with open(self.path) as file:
            first_line=file.readline()
            spliter=check_spliter(first_line)

        with open(self.path) as file:
            if haveHead:
                first_line = file.readline()
                self.head=first_line.strip().split(spliter)

            dataset = []
            for line in file.readlines():
                lineArr = line.strip().split(spliter)
                if isLabeled:
                    # 如果有标签的数据集，标签一般在最后一列
                    # 对每个标签统计个数，并用数字0123代替该标签字符串
                    self.label.append(lineArr[-1])
                    lineArr=lineArr[:-1]
                dataset.append(lineArr)
            self.data = np.array(dataset, dtype=np.float64)
            # print("该数据集有%d个%d维数据" % (data.shape[0], data.shape[1]))
        return self.data

    def write_data(self):
        """
        将numpy_ndarray输出到txt文件
        """
        if self.label==[]:
            np.savetxt(self.path,self.data,delimiter='\t')
        else:
            np.savetxt(self.path, np.column_stack((self.data, self.label)), delimiter='\t',
                       fmt=['%f'] * self.dimension + ['%d'])
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

    def draw_2D_data(self,x=0,y=1):
        """
        用matplotlib绘制数据
        :param x:x轴默认是第0列的数据
        :param y:y轴默认是第1列的数据
        """
        self.show_shape()
        assert x<self.dimension and y<self.dimension

        if self.label==[]:
            # 无标签数据集可视化
            # 可视化一下
            plt.scatter(self.data[:, x], self.data[:, y], marker='o', color='r')
        else:
            # 有标签数据集可视化
            # 每个标签scatter一次，按label_count中label的顺序（即最多标签的优先scatter）
            new_label=self.label_to_nparray()
            for i in range(len(self.label_count)):
                plt.scatter(self.data[new_label==i][:, x],self.data[new_label==i][:, y],marker='o')

            #设置图例
            legend_title=self.head[-1] if self.head else None
            plt.legend(self.label_count.keys(),title=legend_title)

        # 如果数据集有头，那么xy轴就有意义，绘制时要添加说明
        if self.head!=[]:
            plt.xlabel(self.head[x])
            plt.ylabel(self.head[y])
        else:
            plt.xlabel('x0')
            plt.ylabel('x1')
        plt.show()

    def label_to_nparray(self)->np.ndarray:
        """
        标签数字化，用np.array代替label字符串标签
        :return:np.array
        """
        if isinstance(self.label,np.ndarray):
            return self.label

        ans=[]
        list_label_int = list(self.label_count.keys())
        for i in self.label:
            ans.append(list_label_int.index(i))
        return np.array(ans)

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
    new_set.label=y1
    return new_set

def rand_circles_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_circles(n_samples=n,noise=0.07,random_state=16,factor=0.5)
    new_set.data=x1
    new_set.label=y1
    return new_set

def rand_blob_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_blobs(n_samples=n,n_features=2,centers=3,random_state=170)
    new_set.data=x1
    new_set.label=y1
    return new_set

def rand_bars_data(n:int)->Dataset:
    new_set = Dataset()
    x1,y1=make_blobs(n_samples=n,n_features=2,centers=3,random_state=170)
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    x1 = np.dot(x1, transformation)
    new_set.data=x1
    new_set.label=y1
    return new_set


def check_spliter(line: str) -> str:
    SPLITER = ['\t', ',']
    for i in SPLITER:
        lineArr = line.strip().split(i)
        if len(lineArr) > 1:
            return i
    return ''
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

    path='knn_code/testdata/moonSet.txt'
    new_set=rand_moon_data(300)
    new_set.path=path
    new_set.draw_2D_data()
    new_set.write_data()

    # path='knn_code/testdata/IRIS.csv'
    # new_dataset=Dataset(path=path,isLabeled=True,haveHead=True)
    # new_dataset.draw_2D_data(x=1,y=2)

if __name__ == '__main__':
    main()