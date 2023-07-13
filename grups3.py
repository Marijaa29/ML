#Change the number of clusters, and graphically display the dependence of the criterion function on the number of clusters.
#Based on the graphic representation, determine the optimal number of clusters.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Loading dataset
data = pd.read_csv('data/Mall_Customers.csv')

#Selecting the features 
X = data[['Spending_Score', 'Annual_Income_(k$)']]

#Selectin maximum number of clusters to consider
max_clusters = 10

#Fitting model and data into inertia
inertias = []

for num_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)

    inertia = kmeans.inertia_   #metric used to evaluate the quality of the clustering results
    inertias.append(inertia)

#Here, an empty list inertias is created to store the calculated inertia values. 
# The code then loops through the range of cluster numbers from 1 to max_clusters. 
# For each number of clusters, a K-means model is fitted to the data and the inertia 
# (sum of squared distances of samples to their closest cluster center) is calculated using kmeans.inertia_.
# The inertia value is then appended to the inertias list.


#Graphical display
plt.plot(range(1, max_clusters + 1), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Dependence of inertia on the number of clusters')
plt.show()
