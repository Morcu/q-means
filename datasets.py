import numpy as np
from sklearn import datasets as skdata
import pandas as pd
from sklearn import preprocessing
from scipy.cluster.vq import whiten

def generate_dataset(k=3, n_data_points_per_class=50):
    """
    Generates random dataset
    """

    n = k * n_data_points_per_class
    
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


def load_dataset(name="toy"):
    ##############################
    # Random Dataset
    ##############################
    if name == "random":
        data, category = generate_dataset()


    ##############################
    # Iris Dataset
    ##############################
    elif name == "iris":
        iris = skdata.load_iris()
        data = whiten(iris.data[:, :2])  # we only take the first two features.
        category = iris.target


    ##############################
    # Blob Dataset
    ##############################
    elif name == "blob":
        data = skdata.make_blobs(n_samples=200, n_features=2,
                                centers=3, cluster_std=1.8,random_state=101)
        data, category = data[0], data[1]


    ##############################
    # Original Dataset
    ##############################
    elif name == "toy":
        # Get the data from the .csv file
        df = pd.read_csv('kmeans_data.csv',
            usecols=['Feature 1', 'Feature 2', 'Class'])
        df["Class"] = pd.Categorical(df["Class"])
        df["Class"] = df["Class"].cat.codes
        data = df.values[:, 0:2]
        category = df.values[:, 2].astype(np.int64)


    ##############################
    # Scale data
    ##############################
    data = preprocessing.maxabs_scale(data)

    return data, category