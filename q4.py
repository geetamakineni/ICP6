# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:17:05 2020

@author: geeta
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')
dataset.isna().sum()
dataset.fillna(dataset.mean(),inplace=True)
dataset.isna().sum()

X = dataset.iloc[:,1:17]
Y = dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)

from sklearn.decomposition import PCA# Make an instance of the Model
pca= PCA(4)
X_pca= pca.fit_transform(X_scaled)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(X_pca)
y_cluster_kmeans = km.predict(X_pca)

from sklearn import metrics
score = metrics.silhouette_score(X_pca, y_cluster_kmeans)
print('Silhouette score for',3,'clusters after scaled',score)