# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:27:25 2023

@author: mecheste
"""

# 1
import numpy as np
import scipy.linalg as la

def is_banded(matrix, p, q):
    """
    Check if a matrix is p; q-banded.
    """
    n, m = matrix.shape
    for i in range(n):
        for j in range(m):
            if abs(i - j) > p and abs(j - i) > q and matrix[i][j] != 0:
                return False
    return True

def main():
    # Define matrix size and bands p and q
    n = 6  # Matrix size
    p = 2  # Upper band width
    q = 1  # Lower band width

    num_matrices = 5  # Number of random matrices to generate

    for _ in range(num_matrices):
        # Generate a random matrix with uniformly distributed coefficients in [0, 1]
        A = np.random.rand(n, n)

        # Make the matrix p; q-banded (zero out elements outside the bands)
        for i in range(n):
            for j in range(n):
                if abs(i - j) > p or abs(j - i) > q:
                    A[i][j] = 0

        # Compute the LU factorization of A
        P, L, U = la.lu(A)

        # Compute the PA = LU factorization
        PA = np.dot(P, A)
        P, L2, U2 = la.lu(PA)

        # Analyze the banded structure of L + U for both cases
        is_banded_A_LU = is_banded(L + U, p, q)
        is_banded_PA_LU = is_banded(L2 + U2, p, q)

        print("Matrix A:")
        print(A)
        print("Is A p; q-banded for LU factorization?", is_banded_A_LU)
        print("Is A p; q-banded for PA = LU factorization?", is_banded_PA_LU)
        print("=" * 40)

if __name__ == "__main__":
    main()

    
    
# 2
import numpy as np

def multiply_banded_matrices(A, B, pA, qA, pB, qB):
    """
    Multiply two banded matrices A and B and store the result in a banded matrix C.
    
    Args:
    A (numpy.ndarray): Input banded matrix A of shape (n, n).
    B (numpy.ndarray): Input banded matrix B of shape (n, n).
    pA (int): Left/lower bandwidth of matrix A.
    qA (int): Right/upper bandwidth of matrix A.
    pB (int): Left/lower bandwidth of matrix B.
    qB (int): Right/upper bandwidth of matrix B.
    
    Returns:
    numpy.ndarray: The resulting banded matrix C of shape (n, n).
    """
    n = A.shape[0]
    assert A.shape == B.shape == (n, n)
    
    # Calculate the bandwidth of matrix C
    pC = pA + pB
    qC = qA + qB
    
    # Initialize the result matrix C with zeros
    C = np.zeros((n, n))
    
    # Perform the matrix multiplication and populate C by diagonals
    for i in range(n):
        for j in range(max(0, i - qC), min(n, i + pC + 1)):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(max(0, i - qA), min(n, i + pA + 1)))
    
    return C

# Example usage:
n = 5  # Matrix size
pA = 1  # Left/lower bandwidth of A
qA = 1  # Right/upper bandwidth of A
pB = 1  # Left/lower bandwidth of B
qB = 1  # Right/upper bandwidth of B

# Create banded matrices A and B (for demonstration purposes)
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Ensure A and B are pA; qA and pB; qB banded matrices respectively
for i in range(n):
    for j in range(n):
        if abs(i - j) > pA or abs(j - i) > qA:
            A[i][j] = 0
        if abs(i - j) > pB or abs(j - i) > qB:
            B[i][j] = 0

# Multiply A and B and store the result in C
C = multiply_banded_matrices(A, B, pA, qA, pB, qB)
print("Resulting banded matrix C:")
print(C)



# 3
## a
import numpy as np
from scipy.sparse import csr_matrix

# Define the matrix values
data = [10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5, 8, 9, 9, 13, 4, 2, -1]

# Define the row indices (0-based)
row_indices = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5]

# Define the column indices (0-based)
column_indices = [0, 5, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 1, 4, 5]

# Create the CSR matrix
csr_matrix_example = csr_matrix((data, (row_indices, column_indices)), shape=(6, 6))

# Print the CSR matrix
print(csr_matrix_example)

## b
import numpy as np
from scipy.sparse import csr_matrix

def saxpy_csr(alpha, csr_matrix, x, y):
    """
    Perform the SAXPY operation using CSR format sparse matrix.
    
    Args:
    alpha (float): Scalar multiplier.
    csr_matrix (scipy.sparse.csr_matrix): Sparse matrix in CSR format.
    x (numpy.ndarray): Dense vector x.
    y (numpy.ndarray): Dense vector y.
    
    Returns:
    numpy.ndarray: Resulting vector alpha * A * x + y.
    """
    # Check dimensions
    if csr_matrix.shape[1] != len(x) or csr_matrix.shape[0] != len(y):
        raise ValueError("Matrix-vector dimensions do not match.")

    # Unpack CSR matrix
    data = csr_matrix.data
    indices = csr_matrix.indices
    indptr = csr_matrix.indptr

    # Perform SAXPY operation
    result = np.copy(y)
    for i in range(csr_matrix.shape[0]):
        start_idx = indptr[i]
        end_idx = indptr[i + 1]
        for j in range(start_idx, end_idx):
            result[i] += alpha * data[j] * x[indices[j]]

    return result

# Example usage:
# Create a sparse matrix in CSR format
data = [2.0, 3.0, 1.5, 4.0, 2.5]
indices = [1, 3, 0, 2, 3]
indptr = [0, 2, 2, 4, 5]
csr_matrix_example = csr_matrix((data, indices, indptr), shape=(5, 4))

# Define scalar alpha and vectors x and y
alpha = 2.0
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Perform SAXPY operation
result = saxpy_csr(alpha, csr_matrix_example, x, y)

# Print the result
print("Resulting vector:")
print(result)

## c
import numpy as np
from scipy.sparse import csr_matrix

def csr_transpose_vector_product(A, x):
    """
    Compute the transpose product y = A^T * x for a CSR matrix A and a vector x.
    
    Args:
    A (scipy.sparse.csr_matrix): Sparse matrix A in CSR format.
    x (numpy.ndarray): Dense vector x.
    
    Returns:
    numpy.ndarray: Resulting vector y.
    """
    if A.shape[1] != len(x):
        raise ValueError("Matrix-vector dimensions do not match.")
    
    y = np.zeros(A.shape[0])  # Initialize the result vector y

    # Iterate over rows of A (columns of A^T)
    for i in range(A.shape[1]):
        start_idx = A.indptr[i]
        end_idx = A.indptr[i + 1]
        for j in range(start_idx, end_idx):
            col_idx = A.indices[j]  # Column index of the non-zero element
            y[col_idx] += A.data[j] * x[i]

    return y

# Test program
# Create a sparse matrix A in CSR format
data = [2.0, 3.0, 1.5, 4.0, 2.5]
indices = [1, 3, 0, 2, 3]
indptr = [0, 2, 2, 4, 5]
csr_matrix_example = csr_matrix((data, indices, indptr), shape=(5, 4))

# Define vector x
x = np.array([1.0, 2.0, 3.0, 4.0])

# Compute the transpose product y = A^T * x
result = csr_transpose_vector_product(csr_matrix_example, x)

# Print the result
print("Resulting vector y:")
print(result)



# 4
import numpy as np
import scipy.io
from scipy.sparse.linalg import spsolve

# Step 1: Choose a sparse matrix (e.g., CurlCurl_0) and download the .mtx file
# You can download the .mtx file from the University of Florida Sparse Matrix Collection.

# Step 2: Specify the path to the downloaded .mtx file
matrix_file = "path/to/CurlCurl_0.mtx"  # Update with the actual path

# Step 3: Read the matrix from the .mtx file using scipy
A = scipy.io.mmread(matrix_file)

# Step 4: Generate a random vector 'b' (same size as A's number of rows)
n = A.shape[0]
b = np.random.rand(n)

# Step 5: Solve the linear system Ax = b
x = spsolve(A, b)

# Print the solution vector 'x'
print("Solution vector x:")
print(x)

