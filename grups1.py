"""Demonstration of K-means clustering algorithm from the "Mall_Customers.csv" dataset"""

# Load Mall_Customers.csv data. 
# Using the Spending_score size, and one size of your choice,
# determine the cluster centers using the Kmeans algorithm.

import pandas as pd
from sklearn.cluster import KMeans

#Loading data
data = pd.read_csv('data/Mall_Customers.csv')

#Selecting the features 
X = data[['Spending_Score', 'Annual_Income_(k$)']]

#Selecting the number of clusters
num_clusters = 5

#Performing K-Means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

#Retrieving the cluster centers
cluster_centers = kmeans.cluster_centers_

#Printing the cluster centers
print("Cluster centers:")

for center in cluster_centers:
    print(center)