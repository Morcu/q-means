#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from q_distance import distance_centroids

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

# Setting centers seed
centers = np.array([[-0.25, 0.2], [0, -0.1], [0.25, 0.35]])
print(centers)


def point_centroid_distances(point, centroids):
    
    # Calculating theta and phi values
    phi_list = [((x + 1) * pi / 2) for x in [point[0], centroids[0][0], centroids[1][0], centroids[2][0]]]
    theta_list = [((x + 1) * pi / 2) for x in [point[1], centroids[0][1], centroids[1][1], centroids[2][1]]]

    # Create a 2 qubit QuantumRegister - two for the vectors, and 
    # one for the ancillary qubit
    qreg = QuantumRegister(3, 'qreg')

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(1, 'creg')

    qc = QuantumCircuit(qreg, creg, name='qc')

    # Get backend using the Aer provider
    backend = Aer.get_backend('qasm_simulator')

    # Create list to hold the results
    results_list = []

    # Estimating distances from the new point to the centroids
    for i in range(1, 4):
        # Apply a Hadamard to the ancillary
        qc.h(qreg[2])

        # Encode new point and centroid
        qc.u3(theta_list[0], phi_list[0], 0, qreg[0])           
        qc.u3(theta_list[i], phi_list[i], 0, qreg[1]) 

        # Perform controlled swap
        qc.cswap(qreg[2], qreg[0], qreg[1])
        # Apply second Hadamard to ancillary
        qc.h(qreg[2])

        # Measure ancillary
        qc.measure(qreg[2], creg[0])

        # Reset qubits
        qc.reset(qreg)

        # Register and execute job
        job = execute(qc, backend=backend, shots=1024)
        result = job.result().get_counts(qc)
        try:
            results_list.append(result['1'])
        except:
            results_list.append(0)


    return results_list


threshold = 2e-2
error_tolerance = 1e-1

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)
upper_error = np.inf

# When, after an update, the estimate of that center stays the same, exit loop
while (error - error_tolerance) < upper_error and error > threshold:
    # Measure the distance to every center
    distances = np.array(list(map(lambda x: distance_centroids(x, centers), data)))

    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)

    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
        
    upper_error = deepcopy(error)
    error = np.linalg.norm(centers_new - centers_old)

    print(error)

centers_new


# Plot the data and the centers generated as random
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[clusters[i]])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)

plt.show()
