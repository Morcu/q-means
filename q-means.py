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


class QMeans():

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

        # When, after an update, the estimate of that center stays the same, exit loop
        while error > self.threshold:
            # Measure the distance to every center
            distances = np.array(list(map(lambda x: distance_centroids_parallel(x, centers_new), self.data)))

            # Assign all training data to closest center
            clusters = np.argmin(distances, axis = 1)
            
            centers_old = deepcopy(centers_new)

            # Calculate mean for every cluster and update the center
            for i in range(k):
                if np.sum(clusters == i) != 0:
                    centers_new[i] = np.mean(data[clusters == i], axis=0)
                else:
                    centers_new[i] = np.random.normal(scale=0.6, size=centers_new[i].shape)
                
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
    qmeans = QMeans(data, k)

    qmeans.run()

    qmeans.plot()

    