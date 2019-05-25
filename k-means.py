#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Get the data from the .csv file
df = pd.read_csv('kmeans_data.csv',
    usecols=['Feature 1', 'Feature 2', 'Class'])



def generate_dataset(k=3, n_data_points_per_class=50):
    """
    Generate random dataset
    """
    # Number of clusters
    true_centroids_x = [-0.4, 0.6, 0.0]
    true_centroids_y = [-0.4, 0.0, 0.8]
    true_centroids_var_x = [0.25, 0.2, 0.1]
    true_centroids_var_y = [0.25, 0.2, 0.6]

    x = np.random.normal(loc=true_centroids_x[0], scale=true_centroids_var_x[0], size=n_data_points_per_class) 
    y = np.random.normal(loc=true_centroids_y[0], scale=true_centroids_var_y[0], size=n_data_points_per_class)
    category = np.repeat(0, n_data_points_per_class)
    for i in range(1, k):
        category = np.vstack((category, np.repeat(i, n_data_points_per_class)))
        x = np.vstack((x, np.random.normal(loc=true_centroids_x[i], scale=true_centroids_var_x[i], size=n_data_points_per_class)))
        y = np.vstack((y, np.random.normal(loc=true_centroids_y[i], scale=true_centroids_var_y[i], size=n_data_points_per_class)))

    data = np.vstack([x.reshape(n), y.reshape(n)]).transpose()
    category = category.reshape(n)

    return data, category

k=3
n_data_points_per_class=50
n = k*n_data_points_per_class

# Plot the data and the centers generated as random
colors=['green', 'blue', 'black']
data, category = generate_dataset()


centers = np.array([[-0.25, 0.2], [0, -0.1], [0.25, 0.35]])

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
    print(error)
centers_new 


# Plot the data and the centers generated as random
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[clusters[i]])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)

plt.show()




