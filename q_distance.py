from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from numpy import pi
import numpy as np


def encode_feature(x):
    """"
    We map data feature values to \theta and \phi values using this equation:
        \phi = (x + 1) \frac{\pi}{2},
    where \phi is the phase and \theta the angle
    """
    return ((x + 1) * pi / 2)


def binary_combinations(n):
    """
    Returns all possible combinations of length n binary numbers as strings
    """
    combinations = []
    for i in range(2**n):
        bin_value = str(bin(i)).split('b')[1]
        while len(bin_value) < n:
            bin_value = "0" + bin_value 

        combinations.append(bin_value)

    return combinations


def binary_combinations_pos(n, index):
    """
    Returns all possible combinations of binary numbers where bit index=1
    """
    combinations_pos = []
    for bin_number in binary_combinations(n):
        if bin_number[n - index - 1] == '1':
            combinations_pos.append(bin_number)

    return combinations_pos


def distance_centroids(point, centroids):

    k = len(centroids)
    x_point, y_point = point[0], point[1]

    
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

    # Get backend using the Aer provider
    backend = Aer.get_backend('qasm_simulator')

    
    # for i in range(k):
         

    for i in range(k):
        # Encode the point to measure and centroid
        qc.u3(theta_list[i], phi_list[i], 0, qreg_centroid[i])
        qc.u3(theta_input, phi_input, 0, qreg_input[i])

        # Apply a Hadamard to the ancillaries
        qc.h(qreg_psi[i])

        # Perform controlled swap
        qc.cswap(qreg_psi[i], qreg_input[i], qreg_centroid[i]) 

        # Apply second Hadamard to ancillary
        qc.h(qreg_psi[i])

        # Measure ancillary
        qc.measure(qreg_psi[i], creg[i])


    # Register and execute job
    job = execute(qc, backend=backend, shots=1024)
    result = job.result().get_counts(qc)

    distance_centroids = [0]*k
    for i in range(k):
        keys_centroid_k = binary_combinations_pos(k, i)
        for key in keys_centroid_k:
            if key in result:
                distance_centroids[i] += result[key]
    
    return distance_centroids