# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:39:27 2020

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
for i in range(1,10):
    kmeans= KMeans(n_clusters=i,init='k-means++',max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.grid(b=True, which='major', color='RED', linestyle='-')
plt.show()


##building the model
from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X)
from sklearn import metrics
score = metrics.silhouette_score(X, y_cluster_kmeans)
print(score)