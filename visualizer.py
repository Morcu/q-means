#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
import sys

def encode_feature(x):
    """"
    We map data feature values to \theta and \phi values using this equation:
        \phi = (x + 1) \frac{\pi}{2},
    where \phi is the phase and \theta the angle
    """
    return ((x + 1) * pi / 2)

def main(n_features=2, csv_data='kmeans_data.csv', circuit_file='circuit.png'):


    read_feats = ['Feature ' + str(i) for i in range(1, n_features + 1)]
    read_feats.append("Class")
    # Get the data from the .csv file
    df = pd.read_csv(csv_data, usecols=read_feats)

    colors = df["Class"].unique().tolist()
    colors.reverse()
    # Number of clusters
    k = len(colors)

    data = df.values[:, 0:2]
    category = df.values[:, 2]
    c = data.shape[1]

    #mean = np.mean(data, axis=0)
    #std = np.std(data, axis=0)
    #centers = np.random.randn(k, c) * std + mean

    x_point, y_point = -0.161, 0.141
    centroids = np.array([[-0.25, 0.2], [0, -0.1], [0.25, 0.35]])

    # Calculating theta and phi values
    phi_list = [encode_feature(x) for x in [centroids[0][0], centroids[1][0], centroids[2][0]]]
    theta_list = [encode_feature(x) for x in [centroids[0][1], centroids[1][1], centroids[2][1]]]
    phi_input = encode_feature(x_point)
    theta_input = encode_feature(y_point)

    # We need 3 quantum registers, of size k one for a data point (input),
    # one for each centroid and one for each ancillary
    qreg_input = QuantumRegister(k, name='qreg_input')
    qreg_centroid = QuantumRegister(k, name='qreg_centroid')
    qreg_psi = QuantumRegister(k, name='qreg_psi')

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(k, 'creg')

    # Create the quantum circuit containing our registers
    qc = QuantumCircuit(qreg_input, qreg_centroid, qreg_psi, creg, name='qc')

    backend = Aer.get_backend('qasm_simulator')

    # Encode the point to measure and centroid
    for i in range(k):
        qc.u3(theta_list[i], phi_list[i], 0, qreg_centroid[i])
        qc.u3(theta_input, phi_input, 0, qreg_input[i])

    for i in range(k):
        # Apply a Hadamard to the ancillaries
        qc.h(qreg_psi[i])

        # Perform controlled swap
        qc.cswap(qreg_psi[i], qreg_input[i], qreg_centroid[i])

        # Apply second Hadamard to ancillary
        qc.h(qreg_psi[i])

        # Measure ancillary
        qc.measure(qreg_psi[i], creg[i])
    # Change the background color in mpl

    style = {'backgroundcolor': '#DFEAEC', 'showindex': False, 'displaycolor': {
            'id': 'red', 'meas':'#0066DA', 'h': '#00DAA7', 'u3': '#EFFF1B'} }

    qc.draw(output='mpl', style=style, reverse_bits= False,filename=circuit_file,scale=1.2)


if __name__ == "__main__":

    try:
        n_features = int(sys.argv[1])
    except:
        n_features = 2

    try:
        csv_data = str(sys.argv[2])
    except:
        csv_data = 'kmeans_data.csv'

    try:
        circuit_file = str(sys.argv[3])
    except:
        circuit_file = 'circuit.png'

    main(n_features, csv_data,circuit_file)