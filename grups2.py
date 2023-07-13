#Graphically display the calculated clusters,
# i.e. color individual data depending on its belonging to a certain cluster.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

#Graphical display of the cluster
labels = kmeans.labels_

plt.scatter(X['Spending_Score'], X['Annual_Income_(k$)'], c=labels) #scatter plot where the x-axis represents the "Spending_Score" and the y-axis represents "Annual_Income_(k$)". Each data point is colored based on its assigned cluster label (labels). The c=labels parameter assigns the colors to the data points based on the cluster labels.
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=200) #cluster_centers[:, 0] and cluster_centers[:, 1] retrieve the x and y coordinates of the cluster centers, respectively. The marker='x' parameter specifies that the marker shape should be 'x'. The color='red' parameter sets the color of the markers to red, and the s=200 parameter controls the size of the markers.
plt.xlabel('Spending Score')
plt.ylabel('Annual Income (k$)')
plt.title('K-means Clustering')
plt.show()

# Analyzing the clustering results & Display statistical information for each cluster
data['Cluster'] = kmeans.labels_  # retrieves the assigned cluster labels for each data point in the dataset. These labels represent which cluster each data point belongs to. 


cluster_stats = data.groupby('Cluster').agg({
    'Spending_Score': ['mean', 'median'],
    'Annual_Income_(k$)': ['mean', 'median'],
    'CustomerID': 'count'
})
print(cluster_stats)

#This code adds a new column 'Cluster' to the original dataset,
# containing the assigned cluster labels
# It then calculates the mean, median spending score, annual income,
# and the count of customers for each cluster using groupby and agg functions. 
# The resulting cluster statistics are printed to the console.
