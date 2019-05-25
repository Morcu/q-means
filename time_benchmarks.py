from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from numpy import pi
import numpy as np
import time


def encode_feature(x):
    """"
    We map data feature values to \theta and \phi values using this equation:
        \phi = (x + 1) \frac{\pi}{2},
    where \phi is the phase and \theta the angle
    """
    return ((x + 1) * pi / 2)


def euclidean_distance(point, centroids):
    k = len(centroids)
    
    distances = np.zeros(k)
    for i in range(k):
        distances[i] = np.linalg.norm(point - centroids[i])

    return distances

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


def quantum_distance(point, centroids):

    k = len(centroids)
    
    x_point, y_point = point[0], point[1]
    
    
    # Calculating theta and phi values
    phi_list = []
    theta_list = []
    for i in range(k):
        phi_list.append(encode_feature(centroids[i][0]))
        theta_list.append(encode_feature(centroids[i][1]))

    phi_input = encode_feature(x_point)
    theta_input = encode_feature(y_point)

    # Create a 2 qubit QuantumRegister - two for the vectors, and 
    # one for the ancillary qubit
    qreg = QuantumRegister(3, 'qreg')

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(1, 'creg')

    qc = QuantumCircuit(qreg, creg, name='qc')
    

    backend = Aer.get_backend('qasm_simulator')

    # Create list to hold the results
    results_list = []

    # Estimating distances from the new point to the centroids
    for i in range(k):
        # Apply a Hadamard to the ancillary
        qc.h(qreg[2])

        # Encode new point and centroid
        qc.u3(theta_input, phi_input, 0, qreg[0])          
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
        results_list.append(result['1'])
        
    return results_list


def quantum_distance_efficient(point, centroids):

    k = len(centroids)
    x_point, y_point = point[0], point[1]
    
    
    # Calculating theta and phi values
    phi_list = []
    theta_list = []
    for i in range(k):
        phi_list.append(encode_feature(centroids[i][0]))
        theta_list.append(encode_feature(centroids[i][1]))

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



if __name__ == "__main__":
    # This is the point we need to classify
    point = np.array([0.64556962, 0.79545455])

    # Finding the x-coords of the centroids
    centroids = np.array([[ 1.05843141,  0.24009433],
       [ 0.58724279,  1.34453592],
       [ 1.12053479, -0.58636673]])


    n_iters = 1000

    print("- Quantum_distance_efficient")
    start = time.time()
    for i in range(n_iters): 
        # print(i)
        quantum_distance_efficient(point, centroids)
    print("Time quantum_distance_efficient: {}".format((time.time() - start) / n_iters))

    print("- Quantum_distance")
    start = time.time()
    for i in range(n_iters): 
        # print(i)
        quantum_distance(point, centroids)
    print("Time quantum_distance_efficient: {}".format((time.time() - start) / n_iters))

    print("- Euclidean distance")
    start = time.time()
    for i in range(n_iters): 
        # print(i)
        euclidean_distance(point, centroids)
    print("Time euclidean_distance: {}".format((time.time() - start) / n_iters))


    