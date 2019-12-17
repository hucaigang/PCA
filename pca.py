# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:39:04 2019

@author: 426-2019级-1
"""
import numpy as np

class PCA:
    def __init__(self,filename): #初始化
        self.file=open(filename,"r")
        self.string=self.file.readlines()
        self.row,self.col=len(self.string),6
        self.data=np.zeros((self.row,self.col)) #输入数据矩阵初始化
        self.data_transpose=np.zeros((self.col,self.row)) #转置输入数据矩阵初始化
        self.Cx=np.zeros((min(self.row,self.col),min(self.row,self.col))) #协方差矩阵初始化
        self.eigenvalue=np.zeros((1,min(self.row,self.col)))
        self.eigenvector=np.zeros((min(self.row,self.col),min(self.row,self.col)))
        
    def handle(self): #将输入文件转换为numpy矩阵数据
        for i in range(self.row):
            rows=self.string[i].split(' ')
            for j in range(len(rows)):
                self.data[i][j]=float(rows[j])
    
    def covariance_matrix(self):#求输入数据矩阵转置和协方差矩阵
        self.data_transpose=np.transpose(self.data)
        for i in range(self.col): #做中心化
            sum_col=np.sum(self.data_transpose[i])
            for j in range(self.row):
                self.data_transpose[i][j]-=sum_col/self.row
        temp=np.transpose(self.data_transpose)
        self.Cx=self.data_transpose.dot(temp)/self.col
        
    def eigen_solve(self): #求特征值与特征向量
        self.eigenvalue,self.eigenvector=np.linalg.eig(self.Cx)
        
        
if __name__=="__main__":
    filename=str(input("input the filename"))
    pca=PCA(filename)
    pca.handle()
    pca.covariance_matrix()
    pca.eigen_solve() 
    ##
    print(pca.eigenvalue)
