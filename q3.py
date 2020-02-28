# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:00:14 2020

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
    kmeans= KMeans(n_clusters=i,init='k-means++',max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.grid(b=True, which='major', color='RED', linestyle='-')
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(X_scaled)
y_cluster_kmeans = km.predict(X_scaled)

from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('Silhouette score for',3,'clusters after scaled',score)