# -*- coding: utf-8 -*-
# @Time    : 2022/1/11 11:21
# @Author  : Jiangzhesheng
# @File    : dataset_producer.py
# @Software: PyCharm
# @Description: 1、定义数据集类型2、使用sklearn.datasets生成模拟数据集
from typing import Union, Dict, Any, Optional

import pandas as pd
import matplotlib.figure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *

from number import isNumber
FIG_SIZE=(6,6)
class Dataset:
    # 以IRIS数据集为例，第4列species是离散型字符串，则该字典中存储{4:{Iris-setosa:0,Iris-versicolor:1,Iris-virginica:2}}
    # 即每个离散的类型取值用一个数字表示
    count_dict: Dict[int, pd.Series]

    def __init__(self, path='', data:pd.DataFrame=None, haveHead:bool=None):
        self.path=path
        self.data=data
        #load则label是列表，随机生成则label是np.ndarray
        self.count_dict={}

        self.head=[]
        if self.path!='':
            self.load_data(haveHead=haveHead)

    @property
    def size(self):
        return self.data.shape[0]
    @property
    def dimension(self):
        return self.data.shape[1]
    @property
    def numpydata(self):
        return self.data.to_numpy()
    def load_data(self,haveHead=False):
        """
        将txt文件中的数据转换成np数组，存入data属性中，并返回data属性值
        :param filename:模拟数据文件路径
        :return:np数组
        """
        #预读取加载信息
        spliter,haveHead,isstr=self.check_head()

        with open(self.path) as file:
            dataset = []
            if haveHead:
                file.readline()
            for line in file.readlines():
                lineArr = line.strip().split(spliter)
                dataset.append(lineArr)
            self.data=pd.DataFrame(data=dataset,columns=self.head)
            self.data=self.data.astype(isstr)

    def preprocess_data(self):
        for i in range(self.dimension):
            if self.data.dtypes[i]=='object':
                property_name=self.data.columns[i]
                clst: pd.Series=self.data[property_name].value_counts()
                self.count_dict[i]=clst
                replace_dict={clst.index[i]:i for i in range(len(clst))}
                self.data[property_name]=self.data[property_name].replace(replace_dict)

    def write_data(self):
        """
        将numpy_ndarray输出到txt文件
        """
        np.savetxt(self.path,self.data.to_numpy(),delimiter='\t')

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

    def show_shape(self):
        print ("该数据集有%d个%d维数据"%(self.size,self.dimension))

    def draw_2D_data(self,x=0,y=1,label:int=None)->(plt.figure,plt.axis):
        """
        用matplotlib绘制数据
        :param x:作为x轴的列，默认是第0列的数据
        :param y:作为y轴的列，默认是第1列的数据
        :param label:分类标签列，可以是int、bool、None，False和None代表无标签，True默认最后一列
        """
        self.show_shape()
        assert x<self.dimension and y<self.dimension
        fig: matplotlib.figure.Figure=plt.figure()
        ax: matplotlib.figure.Axes = fig.add_subplot(111)

        pltdata=self.data.to_numpy()

        if label==True:
            label=self.dimension-1
        if label==None or label==False:
            # 无标签数据集
            # 可视化一下
            ax.scatter(pltdata[:, x], pltdata[:, y], marker='x', color='blue')
        else:
            # 有标签数据集可视化
            for i in range(len(self.count_dict[label])):
                #在数据集的label列，选出类型为i的，第xy列，作为xy
                pltx=pltdata[pltdata[:, label] == i][:, x]
                plty=pltdata[pltdata[:,label]==i][:, y]
                ax.scatter(pltx,plty,marker='o')
            #设置图例
            legend_title=self.head[-1] if self.head else None
            ax.legend(self.count_dict[label].keys(), title=legend_title)
        #设置坐标轴
        ax.set_xlabel(self.head[x])
        ax.set_ylabel(self.head[y])
        return fig,ax

    def check_head(self)->(str,bool,[]):
        """
        读取数据集的两行，可以确定：数据集的分隔符、是否有头、每个属性的名字和数据类型
        :return:分隔符、是否有头、数据类型列表
        """
        SPLITER = ['\t', ',']#候选分隔符集合
        #返回值初始化
        spliter=''
        headflag=False
        isstr={}#记录每个属性的数据类型，目前只判断有效数字和字符串 TODO：增加判断类型

        with open(self.path) as file:
            #首先确定分隔符
            first,second=file.readline(),file.readline()
            for i in SPLITER:
                lineArr = first.strip().split(i)
                if len(lineArr) > 1:
                    spliter=i
                    break
            #判断完毕直接切分
            first,second = first.strip().split(spliter),second.strip().split(spliter)
            #然后确定数据集是否有头
            for i in range(len(first)):
                if isNumber(first[i])!=isNumber(second[i]):
                    headflag=True
                    break
            self.head=first if headflag else ['X%d'%i for i in range(len(first))]
            #最后确定每个属性的数据类型
            for i in range(len(second)):
                if isNumber(second[i]):
                    isstr[self.head[i]]='float32'
                else:
                    isstr[self.head[i]] = 'object'
                    self.count_dict[i]={}
        return spliter,headflag,isstr

TMP_FILE='./tempSet.txt'
def rand_uniform_data(n:int, d:int)->Dataset:
    """
    随机生成数据
    :param n: 个数
    :param d: 维度
    :return:
    """
    x=np.random.rand(n, d)
    np.savetxt(TMP_FILE,x,delimiter='\t')
    new_set=Dataset(path=TMP_FILE)
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
    # a=Dataset(path='knn_code/testdata/IRIS.csv')
    # a = Dataset(path='kmeans/testdata/barsSet.txt')
    a=rand_uniform_data(300,2)
    a.preprocess_data()
    a.draw_2D_data()
    plt.show()
    # a.show_data()
    # a.load_data()
    # a.normalize_data()
    # a.write_data()
    # a.show_data()

    # path='knn_code/testdata/moonSet.txt'
    # new_dataset = Dataset(path=path,label=True)
    # rand_data = np.zeros((1,new_dataset.dimension))
    # for i in range(new_dataset.dimension):
    #     low = np.min(new_dataset.data[:, i])
    #     high = np.max(new_dataset.data[:, i])
    #     # 在该维度下的最小值到最大值中取一个随机数
    #     rand_data[0][i] = (high - low) * np.random.sample() + low
    # new_dataset.data=np.append(new_dataset.data, rand_data, axis=0)
    # new_dataset.label.append('testpoint')
    # new_dataset.draw_2D_data()

if __name__ == '__main__':
    main()