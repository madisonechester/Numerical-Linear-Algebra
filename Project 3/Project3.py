import os
import tarfile
import numpy as np
from time import time
from scipy.io import mmread
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, issparse

# Exercise 1

def extract_dataset(file_path, extraction_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extraction_path)

def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

def compute_pagerank(G, m=0.15, tol=1e-6, max_iter=100):
    n = G.shape[0]

    # ensure G is a CSR matrix
    if not issparse(G):
        G = csr_matrix(G)

    # compute out-degrees
    out_degrees = np.array(G.sum(axis=1)).flatten()

    # handle zero out-degrees
    out_degrees[out_degrees == 0] = 1

    # identify dangling nodes
    dangling_nodes = (out_degrees == 0)

    # initialize variables
    e = np.ones(n)

    # handle division by zero for non-dangling nodes
    z_non_dangling = np.divide(m, out_degrees, where=(out_degrees != 0), out=np.zeros_like(out_degrees))

    # handle division by zero for dangling nodes
    z_dangling = np.divide(1, n, where=dangling_nodes, out=np.zeros_like(out_degrees))

    # convert to sparse matrix
    z = csr_matrix(z_non_dangling + z_dangling)  
    x = np.ones(n)

    # power method iteration for Mm with handling dangling nodes
    for _ in range(max_iter):
        x_new = (1 - m) * G.dot(x / out_degrees) + m * z @ x

        # normalize the PageRank vector to prevent overflow
        x_new /= np.sum(x_new)

        if np.linalg.norm(x_new - x, 1) < tol:
            break

        x = x_new

    return x_new

# extract the dataset
tar_filename = 'p2p-Gnutella30.tar.gz'
extracted_directory = os.path.splitext(tar_filename)[0]
extract_dataset(tar_filename, extracted_directory)

# list the contents of the extracted directory
list_files(extracted_directory)
print()

# search for the Matrix Market file in the subdirectory
subdirectory = 'p2p-Gnutella30'
matrix_filename = os.path.join(extracted_directory, subdirectory, 'p2p-Gnutella30.mtx')

# check if the file exists before reading
if os.path.exists(matrix_filename):
    # read the link matrix G
    G = mmread(matrix_filename).tocsr()

    # compute PageRank for Mm with damping factor m=0.15
    pagerank_vector = compute_pagerank(G, m=0.15)

    # display the PageRank vector for Mm
    print("PageRank Vector for Mm:\n", pagerank_vector)

    # check if the sum is close to 1
    sum_of_pageranks = np.sum(pagerank_vector)
    print("Sum of PageRank scores:", sum_of_pageranks)
    print()
else:
    print(f"Matrix Market file not found in the expected subdirectory.")

# Exercise 2

def build_diag(G):
    out_degrees = G.sum(axis=1)
    d_ii = np.divide(1, out_degrees, out=np.zeros_like(out_degrees), where=out_degrees!=0)
    return diags(np.squeeze(np.asarray(d_ii)))

def compute_PR_no_store(G, m=0.15, tol=1e-15):
    n = G.shape[0]

    L = [set() for _ in range(n)]
    n_j = np.zeros(n)

    indptr = G.indptr
    for i in range(n):
        L[i] = set(G.indices[indptr[i]:indptr[i + 1]])
        n_j[i] = len(L[i])

    x = np.zeros(n)
    xc = np.ones(n) / n

    while norm(x - xc, np.inf) > tol:
        xc = x.copy()
        x = np.zeros(n)
        for j in range(n):
            if n_j[j] == 0:
                x += xc[j] / n
            else:
                for i in L[j]:
                    x[i] += xc[j] / n_j[j]
        x = (1 - m) * x + m / n

    return x / np.sum(x)

# load the link matrix G
G = mmread(matrix_filename).tocsr()

# build the diagonal matrix D
D = build_diag(G)

# compute A~ = (1 - m)A
A_tilde = (1 - 0.15) * G.dot(D)

# compute the PageRank vector without storing matrices
start = time()
pagerank_no_store = compute_PR_no_store(A_tilde)
end = time()

print("PR vector without storing matrices:\n", pagerank_no_store)
print("Sum of PageRank scores:", np.sum(pagerank_no_store))
print("Execution time:", end - start)
print()

# COMPARISON

# insight into algorithms performance
m_values = np.linspace(0.05, 0.95, num=19)
times_store = []
times_storent = []
precisions = []

# iterate over different damping factors
for m in m_values:
    print("Damping Factor:", m)
    
    # store experiments
    start = time()
    pagerank_store = compute_pagerank(G, m=m)
    end = time()
    times_store.append(end - start)

    # no-store experiments
    start = time()
    pagerank_no_store = compute_PR_no_store(G, m=m)
    end = time()
    times_storent.append(end - start)

    # compare precision between store and no-store implementations
    precision = norm(pagerank_store - pagerank_no_store, 2)
    precisions.append(precision)

    print("- Execution time (with store):", end - start)
    print("- Precision (difference between store and no-store):", precision)

# plot the results
plt.plot(m_values, times_store, color="green", label="With Store")
plt.plot(m_values, times_storent, color="red", label="Without Store")
plt.xlabel('Damping Factor')
plt.ylabel('Seconds')
plt.title('Execution Times for PageRank with and without Matrix Store')
plt.legend()
plt.show()

plt.plot(m_values, precisions, color="blue")
plt.xlabel('Damping Factor')
plt.ylabel('Difference')
plt.title('Precision between PageRank with and without Matrix Store')
plt.show()
