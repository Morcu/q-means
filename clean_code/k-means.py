#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys


def main(n_features=2, csv_data='../kmeans_data.csv'):

    read_feats = ['Feature ' + str(i) for i in range(1, n_features + 1)]
    read_feats.append("Class")
    # Get the data from the .csv file
    df = pd.read_csv(csv_data,
                     usecols=read_feats)

    colors = df["Class"].unique().tolist()
    colors.reverse()
    # Number of clusters
    k = len(colors)

    # Change categorical data to number 0-2
    df["Class"] = pd.Categorical(df["Class"])
    df["Class"] = df["Class"].cat.codes
    # Change dataframe to numpy matrix
    data = df.values[:, 0:n_features]
    category = df.values[:, n_features]

    # Number of training data
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]

    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, c) * std + mean

    # Static data to test
    centers = np.array([[-0.25, 0.2], [0, -0.1], [0.25, 0.35]])

    centers_old = np.zeros(centers.shape)  # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
        print("the error is {}".format(error))


    # Plot the data and the centers generated as random
    for i in range(n):
        plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[int(category[i])])
    plt.scatter(centers_new[:, 0], centers_new[:, 1], marker='*', c='g', s=150)
    plt.show()


if __name__ == "__main__":

    try:
        n_features = int(sys.argv[1])
    except:
        n_features = 2

    try:
        csv_data = str(sys.argv[2])
    except:
        csv_data = '../kmeans_data.csv'

    main(n_features, csv_data)



