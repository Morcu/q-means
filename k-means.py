
#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from q_distance import distance_centroids_parallel


import time
from datasets import load_dataset


class KMeans():

    def __init__(self, trainig_input, k, centroids_ini=None, threshold=0.04, seed=0):
        np.random.seed(seed)

        if centroids_ini is None:
            # Setting centers seed
            self.centers = (np.random.normal(scale=0.6, size=[k, 2]))
        else:
            self.centers = centroids_ini


        self.data = trainig_input
        self.k = k
        self.n = self.data.shape[0]
        self.threshold = threshold


    def run(self):
        """
        Traininig process
        """
        centers_old = np.zeros(self.centers.shape) # to store old centers
        centers_new = deepcopy(self.centers) # Store new centers

        clusters = np.zeros(self.n)
        distances = np.zeros((self.n, self.k))

        error = np.inf

        while error != 0:
            # Measure the distance to every center
            for i in range(self.k):
                distances[:,i] = np.linalg.norm(self.data - self.centers[i], axis=1)
            # Assign all training data to closest center
            clusters = np.argmin(distances, axis = 1)
            
            centers_old = deepcopy(centers_new)
            # Calculate mean for every cluster and update the center
            for i in range(k):
                centers_new[i] = np.mean(data[clusters == i], axis=0)
            error = np.linalg.norm(centers_new - centers_old)
            print(error)

        self.centers = centers_new
        self.clusters = clusters


    def plot(self):
        """
        Plots the results in training
        """
        colors=['green', 'blue', 'black', 'red']
        # Plot the data and the centers generated as random
        for i in range(self.n):
            plt.scatter(data[i, 0], data[i,1], s=7, color = colors[self.clusters[i]])
        plt.scatter(self.centers[:,0], self.centers[:,1], marker='*', c='g', s=150)

        plt.show()

    def fit(self, point):
        """
        Finds the cluster of a given datapoint
        """

if __name__ == "__main__":

    data, category = load_dataset("toy")

    k = np.max(category) + 1
    qmeans = KMeans(data, k)

    qmeans.run()

    qmeans.plot()

    

