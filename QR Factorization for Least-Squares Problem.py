import numpy as np
import pandas as pd
from scipy.linalg import qr 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Exercise 1
# Load data from "dades.txt"
data = np.loadtxt("dades.txt")

# Separate data into x and y
x = data[:, 0]
y = data[:, 1]

# Reshape x to a 2D array
x = x.reshape(-1, 1)

# Define a range of polynomial degrees to test
degrees = np.arange(1, 20)

# Initialize an array to store Sum of Squared Errors (SSE)
sse = np.zeros(degrees.shape)

# Loop through different polynomial degrees
for i, degree in enumerate(degrees):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x)
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict using the model
    y_pred = model.predict(X_poly)
    
    # Calculate Sum of Squared Errors (SSE)
    sse[i] = np.sum((y_pred - y) ** 2)

# Find the degree with the lowest SSE
best_degree = np.argmin(sse) + 1

# Create polynomial features for the best degree
best_poly_features = PolynomialFeatures(degree=best_degree)
X_best_poly = best_poly_features.fit_transform(x)

# Fit the model with the best degree
best_model = LinearRegression()
best_model.fit(X_best_poly, y)
y_best_pred = best_model.predict(X_best_poly)

# Plot the data and the best polynomial fit
plt.scatter(x, y, label="Data")
plt.plot(x, y_best_pred, label=f"Polynomial (Degree {best_degree})", color="r")
plt.legend()
plt.show()



# Create a range of x values for plotting
x_range = np.linspace(min(x), max(x), 100)

# Create polynomial features for the desired degree (degree 13)
poly_features = PolynomialFeatures(degree=13)
X_poly = poly_features.fit_transform(x_range.reshape(-1, 1))

# Predict using the model
y_pred = best_model.predict(X_poly)

# Plot the data and the polynomial fit of degree 13
plt.scatter(x, y, label="Data")
plt.plot(x_range, y_pred, label=f"Polynomial (Degree {13})", color="r")  # Added placeholder for degree
plt.legend()
plt.show()

# Exercise 2
# Load data from "dades.txt"
data = np.loadtxt("dades.txt")

# Separate data into x and y
x = data[:, 0]
y = data[:, 1]

# Reshape x to a 2D array
x = x.reshape(-1, 1)

# Perform QR factorization of x with economic mode
Q, R = qr(x, mode='economic')

# Solve for the coefficients (x) using back-substitution with R
x = np.linalg.solve(R, np.dot(Q.T, y))

# Print the least squares solution
print("LS Solution (x):", x)

# Exercise 3
# Load data from "dades_regression.csv"
data_reg = pd.read_csv("dades_regression.csv", header=None)

# Separate the input features (A) and the target variable (b)
A = data_reg.iloc[:, :-1].values
b = data_reg.iloc[:, -1].values

# Perform QR factorization of A with full mode and pivoting
Q, R, P = qr(A, mode='full', pivoting=True)

# Determine the rank of A
r = np.linalg.matrix_rank(A)

# Split R into R1 (non-singular upper triangular) and S
R1 = R[:r, :r]

# Split b into c and d
c = b[P][:r]
d = b[P][r:]

# Solve the system R1 * u = c for u
u = np.linalg.solve(R1, c)

# Initialize x with zeros and set the components corresponding to u
x = np.zeros(A.shape[1])
x[P[:r]] = u

# Print the basic solution
print("Basic Solution (x):", x)
