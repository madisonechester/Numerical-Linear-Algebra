# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:26:03 2023

@author: mecheste
"""

import numpy as np
from scipy.linalg import svd

A = np.array([[2, 4],
              [1, 3],
              [0, 0],
              [0, 0]])

# EX1

def svd_decomposition(A):
    # Compute A^TA and AA^T
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    # Compute eigenvalues and eigenvectors of A^TA and AA^T
    eigenvalues_ATA, U = np.linalg.eigh(ATA)
    eigenvalues_AAT, V = np.linalg.eigh(AAT)

    # Sort eigenvalues in descending order
    idx_ATA = np.argsort(eigenvalues_ATA)[::-1]
    idx_AAT = np.argsort(eigenvalues_AAT)[::-1]

    # Arrange eigenvectors accordingly
    U = U[:, idx_ATA]
    V = V[:, idx_AAT]

    # Compute singular values from eigenvalues
    singular_values = np.sqrt(np.abs(eigenvalues_ATA[idx_ATA]))

    # Ensure that U and V are orthogonal matrices
    U, _, _ = np.linalg.svd(U, full_matrices=False)
    V, _, _ = np.linalg.svd(V, full_matrices=False)

    # Construct the SVD decomposition matrix
    S = np.zeros_like(A, dtype=float)
    np.fill_diagonal(S, singular_values)

    # Ensure that singular values are arranged in descending order
    U = U[:, :S.shape[0]]
    V = V[:, :S.shape[1]]

    return U, S, V.T

# Example usage:
U, S, Vt = svd_decomposition(A)
print("U:\n", U)
print("S:\n", S)
print("Vt:\n", Vt)

# EX2

# Compute SVD decomposition using scipy.linalg.svd
U, S, Vt = svd(A)

# S is returned as a 1D array, convert it to a diagonal matrix
S = np.diag(S)

# Display the results
print("U:\n", U)
print("S:\n", S)
print("Vt:\n", Vt)

# EX3

def compute_linear_algebra_properties(A):
    # SVD decomposition
    U, S, Vt = np.linalg.svd(A)

    # (a) Rank of A
    rank_A = np.linalg.matrix_rank(A)

    # (b) 2-norm of A (maximum singular value)
    norm_2_A = np.max(S)

    # (c) Frobenius norm of A
    norm_fro_A = np.linalg.norm(A, 'fro')

    # (d) Condition number k2(A)
    cond_number_A = np.max(S) / np.min(S)

    # (e) Pseudoinverse A+
    A_plus = np.linalg.pinv(A)

    return rank_A, norm_2_A, norm_fro_A, cond_number_A, A_plus

# Compute linear algebra properties using SVD
rank_A, norm_2_A, norm_fro_A, cond_number_A, A_plus = compute_linear_algebra_properties(A)

# Display the results
print("Rank of A:", rank_A)
print("2-norm of A:", norm_2_A)
print("Frobenius norm of A:", norm_fro_A)
print("Condition number k2(A):", cond_number_A)
print("Pseudoinverse A+:\n", A_plus)
