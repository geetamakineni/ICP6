# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:04:46 2020

@author: geeta
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('CC.csv')
dataset.isna().sum()

dataset.fillna(dataset.mean(),inplace=True)
dataset.isna().sum()

X = dataset.iloc[:,1:17]
Y = dataset.iloc[:,-1]

##elbow method to know the number of clusters

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.grid(b=True, which='major', color='RED', linestyle='-')
plt.show()