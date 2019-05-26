from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from numpy import pi
import numpy as np


def _encode_feature(x):
    """"
    We map data feature values to \theta and \phi values using this equation:
        \phi = (x + 1) \frac{\pi}{2},
    where \phi is the phase and \theta the angle
    """
    return ((x + 1) * pi / 2)


def _binary_combinations(n):
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


def _binary_combinations_pos(n, index):
    """
    Returns all possible combinations of binary numbers where bit index=1
    """
    combinations_pos = []
    for bin_number in _binary_combinations(n):
        if bin_number[n - index - 1] == '1':
            combinations_pos.append(bin_number)

    return combinations_pos


def distance_centroids_parallel(point, centroids, backend=None, shots=1024):
    """
    Estimates distances using quantum computer specified by backend
    Computes it in parallel for all centroids
    """
    k = len(centroids)
    x_point, y_point = point[0], point[1]
    
    
    # Calculating theta and phi values
    phi_list = []
    theta_list = []
    for i in range(k):
        phi_list.append(_encode_feature(centroids[i][0]))
        theta_list.append(_encode_feature(centroids[i][1]))

    phi_input = _encode_feature(x_point)
    theta_input = _encode_feature(y_point)


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

    if not backend:
        backend = Aer.get_backend('qasm_simulator')  # Get backend using the Aer provider


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
    job = execute(qc, backend=backend, shots=shots)
    result = job.result().get_counts(qc)

    distance_centroids = [0]*k
    for i in range(k):
        keys_centroid_k = _binary_combinations_pos(k, i)
        for key in keys_centroid_k:
            if key in result:
                distance_centroids[i] += result[key]
    
    
    return distance_centroids


def distance_centroids(point, centroids, backend=None, shots=1024):
    """
    Estimates distances using quantum computer specified by backend
    """

    k = len(centroids)

    # Calculating theta and phi values
    phi_list = [_encode_feature(point[0])]
    theta_list = [_encode_feature(point[1])]
    for i in range(k):
        phi_list.append(_encode_feature(centroids[i][0]))
        theta_list.append(_encode_feature(centroids[i][1]))


    # Create a 2 qubit QuantumRegister - two for the vectors, and 
    # one for the ancillary qubit
    qreg = QuantumRegister(k, 'qreg')

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(1, 'creg')

    qc = QuantumCircuit(qreg, creg, name='qc')

    if not backend:
        backend = Aer.get_backend('qasm_simulator')  # Get backend using the Aer provider

    # Create list to hold the results
    results_list = []

    # Estimating distances from the new point to the centroids
    for i in range(1, k + 1):
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
        job = execute(qc, backend=backend, shots=shots)
        result = job.result().get_counts(qc)
        try:
            results_list.append(result['1'])
        except:
            results_list.append(0)


    return results_list