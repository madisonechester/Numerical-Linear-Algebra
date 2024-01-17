# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:23:28 2023

@author: mecheste
"""

# 1 
## a 
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt

# Function to calculate the condition number for Vandermonde matrix of order n
def condition_number_vandermonde(n):
    # Create the Vandermonde matrix Vn
    Vn = np.vander(np.linspace(1, -1, n), increasing=True)
    
    # Calculate the condition number using matrix 2-norm
    cond_num = norm(Vn, 2) * norm(np.linalg.inv(Vn), 2)
    
    return cond_num

# Values of n to consider (you can change this as needed)
n_values = np.arange(2, 51)

# Calculate condition numbers for different values of n
condition_numbers = [condition_number_vandermonde(n) for n in n_values]

# Plot the results
plt.plot(n_values, condition_numbers, marker='o')
plt.xlabel('Matrix Order (n)')
plt.ylabel('Condition Number (k2)')
plt.title('Condition Number of Vandermonde Matrix vs. n')
plt.grid(True)
plt.show()

## b
import numpy as np
from scipy.linalg import hilbert, norm
import matplotlib.pyplot as plt

# Function to calculate the condition number for Hilbert matrix of order n
def condition_number_hilbert(n):
    # Create the Hilbert matrix Hn
    Hn = hilbert(n)
    
    # Calculate the condition number using matrix 2-norm
    cond_num = norm(Hn, 2) * norm(np.linalg.inv(Hn), 2)
    
    return cond_num

# Values of n to consider (you can change this as needed)
n_values = np.arange(2, 21)  # Consider values from 2 to 20

# Calculate condition numbers for different values of n
condition_numbers = [condition_number_hilbert(n) for n in n_values]

# Expected condition numbers according to Szegő's result
expected_condition_numbers = [np.exp(3.5 * n) for n in n_values]

# Plot the results
plt.plot(n_values, condition_numbers, marker='o', label='Computed k2(Hn)')
plt.plot(n_values, expected_condition_numbers, linestyle='--', label='Expected e^(3.5n)')
plt.xlabel('Matrix Order (n)')
plt.ylabel('Condition Number (k2)')
plt.title('Condition Number of Hilbert Matrix vs. n')
plt.legend()
plt.grid(True)
plt.show()



# 2
import numpy as np
from scipy.linalg import norm, solve

# Function to compute the Frank matrix A for a given n
def frank_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j < i - 1:
                A[i, j] = 0.0
            elif j == i - 1:
                A[i, j] = n + 1 - i
            else:
                A[i, j] = n + 1 - j
    return A

# Values of n to consider
n_values = np.arange(2, 25)

# Initialize lists to store condition numbers and relative errors
condition_numbers = []
relative_errors = []

# Compute condition numbers and relative errors for different n
for n in n_values:
    A = frank_matrix(n)
    x0 = np.ones(n)
    b = np.dot(A, x0)
    
    # Calculate condition number using matrix 2-norm
    cond_num = norm(A, 2) * norm(np.linalg.inv(A), 2)
    condition_numbers.append(cond_num)
    
    # Solve the linear system
    x = solve(A, b)
    
    # Calculate the relative error
    rel_err = norm(x - x0, 2) / norm(x0, 2)
    relative_errors.append(rel_err)

# Machine precision (assuming double precision)
machine_precision = 1e-16

# Compare condition numbers with relative errors
for n, cond_num, rel_err in zip(n_values, condition_numbers, relative_errors):
    print(f'n = {n}:')
    print(f'  k2(A) = {cond_num}')
    print(f'  Machine Precision * k2(A) = {machine_precision * cond_num}')
    print(f'  Relative Error of Solution erel(x) = {rel_err}\n')

    
    
# 3
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import time

# Function to generate random matrices with a given condition number
def generate_random_matrix(n, condition_number):
    B = np.random.randn(n, n)  # Random matrix with standard Gaussian coefficients
    Q, _ = np.linalg.qr(B)  # QR decomposition of B
    D = np.diag([condition_number] + [1] * (n - 1))
    A = np.dot(Q, np.dot(D, Q.T))
    return A

# Function to compute the relative backward error
def relative_backward_error(A, x, b):
    r = np.dot(A, x) - b
    rel_error = np.linalg.norm(r) / (np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b))
    return rel_error

# Define dimensions and number of systems to solve
dimensions = range(2, 101)  # n values from 2 to 100
num_systems = 1000  # Number of systems to solve for each dimension

# Initialize lists to store results
min_errors_matrix_inverse = []
max_errors_matrix_inverse = []
min_errors_lu_gepp = []
max_errors_lu_gepp = []
matrix_inverse_times = []
lu_gepp_times = []

# Perform the experiment for each dimension
for n in dimensions:
    condition_number = 9e7  # Condition number k2(A)
    errors_matrix_inverse = []
    errors_lu_gepp = []
    matrix_inverse_execution_time = 0
    lu_gepp_execution_time = 0
    
    for _ in range(num_systems):
        A = generate_random_matrix(n, condition_number)
        x = np.random.randn(n)
        b = np.dot(A, x)
        
        # Solve using matrix inverse method
        start_time = time.time()
        x_inv = np.linalg.solve(A, b)
        matrix_inverse_execution_time += time.time() - start_time
        error_inv = relative_backward_error(A, x_inv, b)
        errors_matrix_inverse.append(error_inv)
        
        # Solve using LU method (GEPP)
        start_time = time.time()
        lu, piv = lu_factor(A)
        x_lu = lu_solve((lu, piv), b)
        lu_gepp_execution_time += time.time() - start_time
        error_lu_gepp = relative_backward_error(A, x_lu, b)
        errors_lu_gepp.append(error_lu_gepp)
    
    min_errors_matrix_inverse.append(min(errors_matrix_inverse))
    max_errors_matrix_inverse.append(max(errors_matrix_inverse))
    min_errors_lu_gepp.append(min(errors_lu_gepp))
    max_errors_lu_gepp.append(max(errors_lu_gepp))
    matrix_inverse_times.append(matrix_inverse_execution_time)
    lu_gepp_times.append(lu_gepp_execution_time)

# Print and compare results
print("Dimensions (n):", dimensions)
print("Minimum Relative Errors (Matrix Inverse):", min_errors_matrix_inverse)
print("Maximum Relative Errors (Matrix Inverse):", max_errors_matrix_inverse)
print("Minimum Relative Errors (LU GEPP):", min_errors_lu_gepp)
print("Maximum Relative Errors (LU GEPP):", max_errors_lu_gepp)

# Plot computation times
import matplotlib.pyplot as plt

plt.plot(dimensions, matrix_inverse_times, label='Matrix Inverse')
plt.plot(dimensions, lu_gepp_times, label='LU GEPP')
plt.xlabel('Dimension (n)')
plt.ylabel('Computation Time (s)')
plt.legend()
plt.title('Computation Time vs. Dimension')
plt.grid(True)
plt.show()



# 4
## a 
import numpy as np

def hager_norm_estimation(A, tol=1e-6, max_iterations=2):
    n = A.shape[0]
    x = np.ones(n)
    y = np.ones(n)
    
    for _ in range(max_iterations):
        z = np.abs(np.dot(A.T, x))
        w = np.abs(np.dot(A, y))
        x = np.sign(z)
        y = np.sign(w)
    
    norm_estimation = np.dot(y, np.dot(A, x))
    
    return norm_estimation

# Example usage:
# Create a random matrix A
n = 5
A = np.random.rand(n, n)

# Estimate the 1-norm of A using Hager's algorithm
estimated_norm = hager_norm_estimation(A)

# Display the result
print("Estimated 1-Norm of A:", estimated_norm)

## b 
import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm
import time

def estimate_condition_number(A, tol=1e-6, max_iterations=2):
    n = A.shape[0]
    
    # Initialize vectors x, y, and ones vector
    x = np.ones(n)
    y = np.ones(n)
    ones = np.ones(n)
    
    # Estimate w by solving the linear system Aw = x
    for _ in range(max_iterations):
        z = np.abs(np.dot(A.T, x))
        w = np.abs(np.dot(A, y))
        x = np.sign(z)
        y = np.sign(w)
    
    # Estimate z by solving the linear system ATz = ones
    lu, piv = lu_factor(A.T)
    z = lu_solve((lu, piv), ones)
    
    # Calculate the 1-norm of w and z
    norm_w = norm(w, 1)
    norm_z = norm(z, 1)
    
    # Calculate the estimated condition number
    condition_number_estimation = norm_w / norm_z
    
    return condition_number_estimation

# Function to generate random matrices with a specified condition number range
def generate_random_matrix_with_condition_number(n, min_condition_number, max_condition_number):
    D = np.diag(np.random.uniform(min_condition_number, max_condition_number, n))
    Q = np.random.randn(n, n)
    A = np.dot(Q, D.dot(Q.T))
    return A

# Define parameters
min_condition_number = 1e3
max_condition_number = 1e16
num_matrices = 1000
dimensions = range(10, 91)

# Initialize lists to store results
estimated_condition_numbers = []
direct_condition_numbers = []

# Perform the experiment for each dimension and matrix
for n in dimensions:
    for _ in range(num_matrices):
        A = generate_random_matrix_with_condition_number(n, min_condition_number, max_condition_number)
        
        # Estimate the condition number
        estimated_condition = estimate_condition_number(A)
        estimated_condition_numbers.append(estimated_condition)
        
        # Calculate the condition number directly (inverting the matrix and computing the norms)
        inv_A = np.linalg.inv(A)
        direct_condition = norm(A, 1) * norm(inv_A, 1)
        direct_condition_numbers.append(direct_condition)

# Compute the ratio between the estimated condition number and the direct computation
condition_number_ratios = [est / direct for est, direct in zip(estimated_condition_numbers, direct_condition_numbers)]

# Display the average ratio
average_ratio = np.mean(condition_number_ratios)
print(f"Average Ratio between Estimated and Direct Condition Numbers: {average_ratio:.6f}")
