# Numerical Linear Algebra - Project 2
# December 10th, 2023 at 11:55PM
# Madison Chester 

# 1. Least Squares Problem 

import numpy as np
from scipy.linalg import svd, qr, solve_triangular, norm
from numpy import genfromtxt, vstack
from numpy import argmin

# read data from a file
def read_data(filename):
    with open(filename, 'r') as file:
        data = np.loadtxt(file)
    return data

# solve the least squares problem using SVD
def svd_LS(A, b):
    U, Sigma, VT = svd(A, full_matrices=False)
    Sigma_inv = np.diag(1 / Sigma)
    x_svd = VT.T @ Sigma_inv @ U.T @ b
    return x_svd

# solve the least squares problem using QR (with pivoting if rank-deficient)
def qr_LS(A, b):
    Rank = np.linalg.matrix_rank(A)

    # full-rank case
    if Rank == A.shape[1]: 
        Q_fullr, R_fullr = np.linalg.qr(A)
        y_aux = np.transpose(Q_fullr).dot(b)
        x_qr = solve_triangular(R_fullr, y_aux)
    else:  # rank-deficient case
        Q, R, P = qr(A, mode='economic', pivoting=True)
        R_def = R[:Rank, :Rank]
        c = np.transpose(Q).dot(b)[:Rank]
        u = solve_triangular(R_def, c)
        v = np.zeros((A.shape[1] - Rank))
        x_qr = np.linalg.solve(np.transpose(np.eye(A.shape[1])[:, P]), np.concatenate((u, v)))

    return x_qr

# calculate the error and norm
def calculate_error_and_norm(A, b, x):
    residuals = np.linalg.norm(A @ x - b)
    solution_norm = np.linalg.norm(x)
    return residuals, solution_norm

# select the best degree based on SVD error
def select_best_degree_svd(degrees, data_func):
    errors = []
    for degree in degrees:
        A, b = data_func(degree)
        x_svd = svd_LS(A, b)
        error, _ = calculate_error_and_norm(A, b, x_svd)
        errors.append(error)

    best_degree_index = np.argmin(errors)
    best_degree = degrees[best_degree_index]
    return best_degree

# select the best degree based on QR error
def select_best_degree_qr(degrees, data_func):
    errors = []
    for degree in degrees:
        A, b = data_func(degree)
        x_qr = qr_LS(A, b)
        error, _ = calculate_error_and_norm(A, b, x_qr)
        errors.append(error)

    best_degree_index = np.argmin(errors)
    best_degree = degrees[best_degree_index]
    return best_degree

# load datasets
def datafile(degree):
    data = genfromtxt("data\dades.txt", delimiter="   ")
    points, b = data[:, 0], data[:, 1]
    A = vstack([points ** d for d in range(degree)]).T
    return A, b

def datafile2(degree):
    data = genfromtxt('data\dades_regressio.csv', delimiter=',')
    A, b = data[:, :-1], data[:, -1]
    return A, b

# main code
degrees = range(3, 10)

# select best degree using SVD for datafile
best_degree_svd_datafile = select_best_degree_svd(degrees, datafile)
print("Best degree using SVD for datafile:", best_degree_svd_datafile)

# solve least squares using SVD for datafile
A_svd_datafile, b_svd_datafile = datafile(best_degree_svd_datafile)
x_svd_datafile = svd_LS(A_svd_datafile, b_svd_datafile)
error_svd_datafile, norm_svd_datafile = calculate_error_and_norm(A_svd_datafile, b_svd_datafile, x_svd_datafile)
print(f"LS solution using SVD for datafile (Degree {best_degree_svd_datafile}):")
print(x_svd_datafile)
print("Error using SVD for datafile:", error_svd_datafile)
print("Norm of the solution using SVD for datafile:", norm_svd_datafile)
print("Polynomial Coefficients:")
print(np.flip(x_svd_datafile)[:best_degree_svd_datafile + 1])
print()

# select best degree using QR for datafile
best_degree_qr_datafile = select_best_degree_qr(degrees, datafile)
print("Best degree using QR for datafile:", best_degree_qr_datafile)

# solve least squares using QR for datafile
A_qr_datafile, b_qr_datafile = datafile(best_degree_qr_datafile)
x_qr_datafile = qr_LS(A_qr_datafile, b_qr_datafile)
error_qr_datafile, norm_qr_datafile = calculate_error_and_norm(A_qr_datafile, b_qr_datafile, x_qr_datafile)
print(f"LS solution using QR for datafile (Degree {best_degree_qr_datafile}):")
print(x_qr_datafile)
print("Error using QR for datafile:", error_qr_datafile)
print("Norm of the solution using QR for datafile:", norm_qr_datafile)
print("Polynomial Coefficients:")
print(np.flip(x_qr_datafile)[:best_degree_qr_datafile + 1])
print()

# select best degree using SVD for datafile2
best_degree_svd_datafile2 = select_best_degree_svd(degrees, datafile2)
print("Best degree using SVD for datafile2:", best_degree_svd_datafile2)

# solve least squares using SVD for datafile2
A_svd_datafile2, b_svd_datafile2 = datafile2(best_degree_svd_datafile2)
x_svd_datafile2 = svd_LS(A_svd_datafile2, b_svd_datafile2)
error_svd_datafile2, norm_svd_datafile2 = calculate_error_and_norm(A_svd_datafile2, b_svd_datafile2, x_svd_datafile2)
print(f"LS solution using SVD for datafile2 (Degree {best_degree_svd_datafile2}):")
print(x_svd_datafile2)
print("Error using SVD for datafile2:", error_svd_datafile2)
print("Norm of the solution using SVD for datafile2:", norm_svd_datafile2)
print("Polynomial Coefficients:")
print(np.flip(x_svd_datafile2)[:best_degree_svd_datafile2 + 1])
print()

# select best degree using QR for datafile2
best_degree_qr_datafile2 = select_best_degree_qr(degrees, datafile2)
print("Best degree using QR for datafile2:", best_degree_qr_datafile2)

# solve least squares using QR for datafile2
A_qr_datafile2, b_qr_datafile2 = datafile2(best_degree_qr_datafile2)
x_qr_datafile2 = qr_LS(A_qr_datafile2, b_qr_datafile2)
error_qr_datafile2, norm_qr_datafile2 = calculate_error_and_norm(A_qr_datafile2, b_qr_datafile2, x_qr_datafile2)
print(f"LS solution using QR for datafile2 (Degree {best_degree_qr_datafile2}):")
print(x_qr_datafile2)
print("Error using QR for datafile2:", error_qr_datafile2)
print("Norm of the solution using QR for datafile2:", norm_qr_datafile2)
print("Polynomial Coefficients:")
print(np.flip(x_qr_datafile2)[:best_degree_qr_datafile2 + 1])
print()

# 2. Graphics Compression

import numpy as np
import imageio
import os
import matplotlib.pyplot as plt

def svd_approximation(image_matrix, rank):
    U, s, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    approx_matrix = np.dot(U[:, :rank], np.dot(np.diag(s[:rank]), Vt[:rank, :]))
    return approx_matrix

def compress_image(input_path, output_folder, ranks):
    image_matrix = imageio.v2.imread(input_path).astype(np.uint8)  # read image and convert to uint8
    original_shape = image_matrix.shape  # original shape of the image

    image_name = os.path.splitext(os.path.basename(input_path))[0]

    # flatten the matrix for Frobenius norm calculation
    flattened_image = image_matrix.flatten()
    norm_full = np.linalg.norm(flattened_image, ord=2)  

    # get the original size and channels of the image
    num_channels = original_shape[-1] if len(original_shape) == 3 else 1

    print(f"Original size of {image_name}: {original_shape}")

    for rank in ranks:
        compressed_channels = []

        for channel in range(num_channels):
            channel_matrix = image_matrix[:, :, channel] if num_channels > 1 else image_matrix
            approx_matrix = svd_approximation(channel_matrix, rank)
            compressed_channels.append(approx_matrix)

        # combine the compressed channels back into the image
        approx_matrix = np.stack(compressed_channels, axis=-1) if num_channels > 1 else compressed_channels[0]

        # flatten the matrix for Frobenius norm calculation
        flattened_approx = approx_matrix.flatten()
        norm_approx = np.linalg.norm(flattened_approx, ord=2)  # Frobenius norm of the compressed image
        compression_ratio = norm_approx / norm_full  # percentage of Frobenius norm captured

        output_filename = f"{image_name}_compressed_rank_{rank}_capture_{compression_ratio:.2f}.jpeg"
        output_path = os.path.join(output_folder, output_filename)
        imageio.imwrite(output_path, approx_matrix.astype(np.uint8))
        print(f"Saved compressed image: {output_path} - Compression Ratio: {compression_ratio:.4f}")

        # plotting the original and compressed images
        plot_images(image_matrix, approx_matrix, image_name, rank, compression_ratio)

def plot_images(original, compressed, image_name, rank, compression_ratio):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(f'Original Image\n{image_name}\nSize: {original.shape}\n')

    plt.subplot(1, 2, 2)
    plt.imshow(compressed)
    plt.title(f'Compressed Image (Rank={rank}, Frob Norm captured={compression_ratio:.2%})\nSize: {compressed.shape}\n')

    plt.tight_layout()
    plt.show()

# create a folder to save compressed images
output_folder = "compressed_images"
os.makedirs(output_folder, exist_ok=True)

image_files = ["sagradafamilia.jpg", "laboqueria.jpg", "montjuic.jpg"]
ranks = [1, 5, 20, 50, 100]  

for image_file in image_files:
    input_path = os.path.join("original_images", image_file)
    compress_image(input_path, output_folder, ranks)
    print()

# 3. Principle Component Analysis 

# example.dat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def compute_PCA_example(file_path='data\example.dat', matrix_type='covariance', print_flag=True):
    # load the dataset
    data = np.loadtxt(file_path)

    # center the data (subtract mean)
    data_centered = data - np.mean(data, axis=0)

    # compute the covariance or correlation matrix based on matrix_type
    if matrix_type == 'covariance':
        matrix = np.cov(data_centered, rowvar=False)
    elif matrix_type == 'correlation':
        matrix = np.corrcoef(data_centered, rowvar=False)
    else:
        raise ValueError("Invalid matrix_type. Use 'covariance' or 'correlation'.")

    # PCA using sklearn
    pca = PCA()
    new_expr = pca.fit_transform(data_centered)

    # total variance explained by each component
    total_var = np.cumsum(pca.explained_variance_ratio_)

    # standard deviation of each principal component
    standar_dev = np.sqrt(pca.explained_variance_)

    # print covariance or correlation matrix
    if print_flag:
        print(matrix)

    # return total variance, standard deviation, PCA coordinates, and the matrix
    return total_var, standar_dev, new_expr, matrix

def Scree_plot_example(S, plot_type, filename):
    # create a new figure for each plot
    plt.figure()

    # compute eigenvalues from the diagonal of the covariance matrix
    eigenvalues = np.linalg.eigvals(S)

    print('\nScree Plot:')
    for i, eig in enumerate(eigenvalues):
        print(f'PC{i+1}: {eig:.4f}')

    # detailed Scree Plot for saving to PNG
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalues')
    plt.title(f'Scree Plot for Example.dat ({plot_type})')

    # save the detailed plot to a file with a unique filename
    plt.savefig(f'exercise3/scree_plot_example_{filename}.png')

# kaiser rule
def Kaiser_example(S):
    return np.sum(np.abs(np.linalg.eigvals(S)) > 1)

# 3/4 rule
def rule_34_example(total_var):
    return np.argmax(total_var >= 0.75) + 1

print('example.dat Analysis\n')

# covariance matrix
print('Covariance Matrix Analysis:')
total_var_example_cov, standar_dev_example_cov, new_expr_example_cov, S_example_cov = compute_PCA_example(matrix_type='covariance')
print()
print('Accumulated total variance in each principal component:')
print(total_var_example_cov)
print()
print('Standard deviation of each principal component:')
print(standar_dev_example_cov)
print()
print('PCA coordinates of original dataset:')
print(new_expr_example_cov)
Scree_plot_example(S_example_cov, 'covariance', 'example_covariance')
print()
print('Kaiser rule:', Kaiser_example(S_example_cov))
print()
print('3/4 rule:', rule_34_example(total_var_example_cov))
print()

# correlation matrix
print('Correlation Matrix Analysis:')
total_var_example_corr, standar_dev_example_corr, new_expr_example_corr, S_example_corr = compute_PCA_example(matrix_type='correlation')
print()
print('Accumulated total variance in each principal component:')
print(total_var_example_corr)
print()
print('Standard deviation of each principal component:')
print(standar_dev_example_corr)
print()
print('PCA coordinates of original dataset:')
print(new_expr_example_corr)
Scree_plot_example(S_example_corr, 'correlation', 'example_correlation')
print()
print('Kaiser rule:', Kaiser_example(S_example_corr))
print()
print('3/4 rule:', rule_34_example(total_var_example_corr))
print()

# RCsGoff.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

# read the CSV file with sample names
def read_csv(file_path='data\RCsGoff.csv'):
    df = pd.read_csv(file_path)
    sample_names = df.columns[1:]  # first column is gene
    return df, sample_names

# compute PCA 
def compute_PCA(print_flag):
    # read data
    data, _ = read_csv()

    # center data
    data_centered = data.iloc[:, 1:] - data.iloc[:, 1:].mean()

    # compute the covariance matrix
    cov_matrix = data_centered.cov()

    # PCA using sklearn
    pca = sklearnPCA()
    new_expr = pca.fit_transform(data_centered)

    # total variance explained by each component
    total_var = np.cumsum(pca.explained_variance_ratio_)

    # standard deviation of each principal component
    standar_dev = np.sqrt(pca.explained_variance_)

    # print covariance matrix
    if print_flag:
        print(cov_matrix)

    # return total variance, standard deviation, PCA coordinates, and S matrix
    return total_var, standar_dev, new_expr, cov_matrix

# generate a Scree Plot
def Scree_plot(S, plot_type, filename):
    # create a new figure for each plot
    plt.figure()

    # plot eigenvalues against the number of components
    eigenvalues = np.linalg.eigvals(S)

    print('\nScree Plot:')
    for i, eig in enumerate(eigenvalues):
        print(f'PC{i+1}: {eig:.4f}')

    # detailed Scree Plot for saving to PNG
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalues')
    plt.title(f'Scree Plot ({plot_type})')

    # save the detailed plot to a file with a unique filename
    plt.savefig(f'exercise3/scree_plot_{filename}.png')

# kaiser rule
def Kaiser(S):
    return np.sum(np.abs(np.linalg.eigvals(S)) > 1)

# 3/4 rule
def rule_34(total_var):
    return np.argmax(total_var >= 0.75) + 1

def generate_file(file_name, new_expr, total_var, sample_names):
    # extract the first 20 samples from new_expr
    X_new = new_expr[:20, :]

    # convert NumPy arrays to DataFrame
    data_df = pd.DataFrame(data=np.concatenate((X_new.T, np.reshape(total_var, (20, 1))), axis=1),
                            index=sample_names,
                            columns=[f"PC{i}" for i in range(1, X_new.shape[1] + 1)] + ["Variance"])

    # save to a text file in the "exercise3" folder
    folder_name = "exercise3"
    file_path = f"{folder_name}/{file_name}"
    data_df.index.name = "Sample"

    # save the file
    data_df.to_csv(file_path)

print('RCsGoff.csv Analysis')

# covariance matrix
print('\nCovariance Matrix:\n')
data_rcsgoff, sample_names_rcsgoff = read_csv()
total_var_rcsgoff, standar_dev_rcsgoff, new_expr_rcsgoff, S_rcsgoff = compute_PCA(1)
print()
print('Accumulated total variance in each principal component:')
print(total_var_rcsgoff)
print()
print('Standard deviation of each principal component:')
print(standar_dev_rcsgoff)
print()
print('PCA coordinates of original dataset:')
print(new_expr_rcsgoff)
Scree_plot(S_rcsgoff, 'covariance', 'rcsgoff_covariance')
print()
print('Kaiser rule:', Kaiser(S_rcsgoff))
print()
print('3/4 rule:', rule_34(total_var_rcsgoff))
print()

# save results to a file
generate_file("rcsgoff_pca.csv", new_expr_rcsgoff, total_var_rcsgoff, sample_names_rcsgoff)