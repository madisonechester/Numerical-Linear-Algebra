import warnings 
warnings.filterwarnings("ignore")

import time
import scipy
import random
import numpy as np
from scipy.linalg import ldl, solve_triangular, cholesky

def calculate_step_size(lambdas, lambda_changes, slacks, slack_changes):
    lambda_indices = [i for i, change in enumerate(lambda_changes) if change < 0]
    slack_indices = [i for i, change in enumerate(slack_changes) if change < 0]
    
    min_lambda_step = min(-lambdas[i] / lambda_changes[i] for i in lambda_indices) if lambda_indices else 1
    min_slack_step = min(-slacks[i] / slack_changes[i] for i in slack_indices) if slack_indices else 1
    
    return min(min_lambda_step, min_slack_step)

def initialize_problem(n):
    np.random.seed(42)
    random.seed(42)

    m = 2 * n
    x, lambd, s = np.zeros(n), np.ones(m), np.ones(m)
    z = np.concatenate((x, lambd, s))
    G = np.identity(n)
    C, d, e, g = np.concatenate((G, -G), axis=1), np.full(m, -10), np.ones(m), np.random.normal(0, 1, n)
    
    return x, lambd, s, z, G, C, d, e, g

def calculate_KKT_matrix(G, C, n, num_constraints, lambd, s):
    
    S_diag, Lambdas_diag = np.diag(s), np.diag(lambd)
    eq1 = np.concatenate((G, -C, np.zeros((n, num_constraints))), axis=1)
    eq2 = np.concatenate((-C.T, np.zeros((num_constraints, num_constraints)), np.identity(num_constraints)), axis=1)
    eq3 = np.concatenate((np.zeros((num_constraints, n)), S_diag, Lambdas_diag), axis=1)
    KKT_matrix = np.concatenate((eq1, eq2, eq3))
    
    return KKT_matrix, S_diag, Lambdas_diag

def objective_function(x, G, g):
    return 0.5 * x @ G @ x + g @ x

def kkt_conditions(x, lambd, s, G, g, C, d):
    
    eq1 = G @ x + g - C @ lambd
    eq2 = s + d - C.T @ x
    eq3 = s * lambd
    
    return np.concatenate((eq1, eq2, eq3))

def solve_problem(n, maxIter=100, epsilon=1e-16, Print_Time="Yes"):
    x, lambd, s, z, G, C, d, e, g = initialize_problem(n)

    if Print_Time == "Yes":
        Start = time.time()

    for i in range(maxIter):
        KKT_matrix, S, Lambdas = calculate_KKT_matrix(G, C, n, 2 * n, lambd, s)
        b = -kkt_conditions(x, lambd, s, G, g, C, d)
        delta = np.linalg.solve(KKT_matrix, b)
        alpha = calculate_step_size(lambd, delta[n:(n + 2 * n)], s, delta[(n + 2 * n):])
        mu = s @ lambd / (2 * n)
        Mu = ((s + alpha * delta[(n + 2 * n):]) @ (lambd + alpha * delta[n:(n + 2 * n)])) / (2 * n)
        sigma = (Mu / mu) ** 3
        b[(n + 2 * n):] = b[(n + 2 * n):] - np.diag(delta[(n + 2 * n):] * delta[n:(n + 2 * n)]) @ e + sigma * mu * e
        delta = np.linalg.solve(KKT_matrix, b)
        alpha = calculate_step_size(lambd, delta[n:(n + 2 * n)], s, delta[(n + 2 * n):])
        z = z + (alpha * delta) * 0.95

        if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:(n + 2 * n)]) < epsilon) or (np.abs(mu) < epsilon):
            break

        x, lambd, s = z[:n], z[n:(n + 2 * n)], z[(n + 2 * n):]
        Mat, S, Lambdas = calculate_KKT_matrix(G, C, n, 2 * n, lambd, s)

    if Print_Time == "Yes":
        End = time.time()

    return End - Start, i, abs(objective_function(x, G, g) - objective_function(-g, G, g)), np.linalg.cond(KKT_matrix)

def M_KKT_LDL(G, C, lamb, s):
    
    S = np.diag(s)
    Lambdas = np.diag(lamb)
    eq1 = np.concatenate((G, -C),axis = 1)
    eq2 = np.concatenate((- np.transpose(C), - np.diag(1 / lamb * s)), axis = 1)
    Mat = np.concatenate((eq1, eq2))
    
    return Mat, S, Lambdas

def solve_ldl(n, maxIter=100, epsilon=1e-16, Print_Time="Yes"):
    x, lamb, s, z, G, C, d, e, g = initialize_problem(n)

    if Print_Time == "Yes":
        Start = time.time()

    for i in range(maxIter):
        Mat, S, Lambda = M_KKT_LDL(G, C, lamb, s)
        lamb_inv = np.diag(1 / lamb)
        b = kkt_conditions(x, lamb, s, G, g, C, d)
        r1 = b[:n]
        r2 = b[n:(n + 2 * n)]
        r3 = b[(n + 2 * n):]
        b = np.concatenate(([-r1, -r2 + (1 / lamb) * r3]))
        L, D, perm = ldl(Mat)
        y = scipy.linalg.solve_triangular(L, b, lower=True, unit_diagonal=True)
        delta = scipy.linalg.solve_triangular(D.dot(np.transpose(L)), y, lower=False)
        deltaS = lamb_inv.dot(-r3 - s * delta[n:])
        delta = np.concatenate([delta, deltaS])
        alpha = calculate_step_size(lamb, (delta[n:(n + 2 * n)]),(s), delta[(n + 2 * n):])
        mu = s.dot(lamb) / (2 * n)
        Mu = ((s + alpha * delta[(n + 2 * n):]).dot(lamb + alpha * delta[n:(n + 2 * n)])) / (2 * n)
        sigma = (Mu / mu) ** 3
        Ds_Dlamb = np.diag(delta[n + 2 * n:] * delta[n:n + 2 * n])
        b = np.concatenate([-r1, -r2 + lamb_inv.dot(r3 + Ds_Dlamb.dot(e) - sigma * mu * e)])
        y = scipy.linalg.solve_triangular(L, b, lower=True, unit_diagonal=True)
        delta = scipy.linalg.solve_triangular(D.dot(np.transpose(L)), y, lower=False)
        deltaS = lamb_inv.dot(-r3 - Ds_Dlamb.dot(e) + sigma * mu * e - s * delta[n:])
        delta = np.concatenate([delta, deltaS])
        alpha = calculate_step_size(lamb, (delta[n:(n + 2 * n)]),(s), delta[(n + 2 * n):])
        z = z + (alpha * delta) * 0.95

        if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:(n + 2 * n)]) < epsilon) or (np.abs(mu) < epsilon):
            break

        x, lamb, s = z[:n], z[n:(n + 2 * n)], z[(n + 2 * n):]
        Mat, S, Lambda = M_KKT_LDL(G, C, lamb, s)

    if Print_Time == "Yes":
        End = time.time()

    return End - Start, i, abs(objective_function(x, G, g) - objective_function(-g, G, g)), np.linalg.cond(Mat)

def M_KKT_Cholesky(G, C, lamb, s):
    
    S = np.diag(s)
    Lambdas = np.diag(lamb)
    Mat = G + C.dot(np.diag(1 / s * lamb)).dot(np.transpose(C))
    
    return Mat, Lambdas, S

def solve_cholesky(n, maxIter=100, epsilon=1e-16, Print_Time="Yes"):
    np.random.seed(42)  
    random.seed(42)  

    m = 2 * n
    x, lamb, s, z, G, C, d, e, g = initialize_problem(n)

    if Print_Time == "Yes":
        Start = time.time()

    Ghat, Lambda, S = M_KKT_Cholesky(G, C, lamb, s)

    for i in range(maxIter):
        S_inv = np.diag(1 / s)
        b = kkt_conditions(x, lamb, s, G, g, C, d)
        r1 = b[:n]
        r2 = b[n:(n + m)]
        r3 = b[(n + m):]
        rhat = -C.dot(np.diag(1 / s)).dot(-r3 + lamb * r2)
        b = -r1 - rhat
        Cholesk = cholesky(Ghat, lower=True)
        y = solve_triangular(Cholesk, b, lower=True)
        delta_x = solve_triangular(np.transpose(Cholesk), y)
        delta_lamb = S_inv.dot(-r3 + lamb * r2) - S_inv.dot(Lambda.dot(np.transpose(C)).dot(delta_x))
        delta_s = -r2 + np.transpose(C).dot(delta_x)
        delta = np.concatenate((delta_x, delta_lamb, delta_s))
        alpha = calculate_step_size(lamb, delta[n:(n + m)], s, delta[(n + m):])
        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[(n + m):]).dot(lamb + alpha * delta[n:(n + m)])) / m
        sigma = (Mu / mu) ** 3
        Ds_Dlamb = np.diag(delta[n + m:] * delta[n:n + m])
        b = -r1 - (-C.dot(np.diag(1 / s)).dot(-r3 - Ds_Dlamb.dot(e) + sigma * mu * e + lamb * r2))
        y = solve_triangular(Cholesk, b, lower=True)
        delta_x = solve_triangular(np.transpose(Cholesk), y)
        delta_lamb = S_inv.dot(-r3 - Ds_Dlamb.dot(e) + sigma * mu * e + lamb * r2) - S_inv.dot(
            lamb * (np.transpose(C).dot(delta_x)))
        delta_s = -r2 + np.transpose(C).dot(delta_x)
        delta = np.concatenate((delta_x, delta_lamb, delta_s))
        alpha = calculate_step_size(lamb, delta[n:(n + m)], s, delta[(n + m):])
        z = z + (alpha * delta) * 0.95

        if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:(n + m)]) < epsilon) or (np.abs(mu) < epsilon):
            break

        x, lamb, s = z[:n], z[n:(n + m)], z[(n + m):]
        Ghat, Lambda, S = M_KKT_Cholesky(G, C, lamb, s)

    if Print_Time == "Yes":
        End = time.time()

    return End - Start, i, abs(objective_function(x, G, g) - objective_function(-g, G, g)), np.linalg.cond(Ghat)

n_values = [10, 30, 50]

for n in n_values:
    print(f"Problem for n={n} using solve_problem:")
    time_problem, iterations_problem, difference_problem, condition_problem = solve_problem(n, Print_Time="Yes")
    print(f"Computation time for solve_problem: {time_problem:.6f} seconds")
    print(f"Iterations needed: {iterations_problem}")
    print(f"Difference from the real minimum: {difference_problem:.6f}")
    print(f"Condition number: {condition_problem:.6f}\n")

    print(f"Problem for n={n} using solve_ldl:")
    time_ldl, iterations_ldl, difference_ldl, condition_ldl = solve_ldl(n, Print_Time="Yes")
    print(f"Computation time for solve_ldl: {time_ldl:.6f} seconds")
    print(f"Iterations needed: {iterations_ldl}")
    print(f"Difference from the real minimum: {difference_ldl:.6f}")
    print(f"Condition number: {condition_ldl:.6f}\n")

    print(f"Problem for n={n} using solve_cholesky:")
    time_cholesky, iterations_cholesky, difference_cholesky, condition_cholesky = solve_cholesky(n, Print_Time="Yes")
    print(f"Computation time for solve_cholesky: {time_cholesky:.6f} seconds")
    print(f"Iterations needed: {iterations_cholesky}")
    print(f"Difference from the real minimum: {difference_cholesky:.6f}")
    print(f"Condition number: {condition_cholesky:.6f}\n")
        
def read_matrix(source, shape, symm=False):
    matrix = np.zeros(shape)
    
    with open(source, "r") as file:
        lines = file.readlines()
        
    for line in lines:
        row, column, value = map(float, line.strip().split())
        matrix[int(row) - 1, int(column) - 1] = value
        if symm:
            matrix[int(column) - 1, int(row) - 1] = value
            
    return matrix

def read_vector(source, n):
    v = np.zeros(n)
    
    with open(source, "r") as file:
        lines = file.readlines()
        
    for line in lines:
        idx, value = map(float, line.strip().split())
        v[int(idx) - 1] = value
        
    return v

def initialize_variables(n, p):
    
    x = np.zeros(n, dtype=np.float64)
    gamma = np.ones(p, dtype=np.float64)
    lamb = np.ones(2 * n, dtype=np.float64)  
    s = np.ones(2 * n, dtype=np.float64) 
    
    return x, gamma, lamb, s

def M_KKT_C5(G, C, A, n, m, p, lamb, s):
    
    S = np.diag(s)
    Lambda = np.diag(lamb)
    temp1 = np.concatenate((G, -A, -C, np.zeros((n, m))), axis=1)
    temp2 = np.concatenate((-np.transpose(A), np.zeros((p, p + 2 * m))), axis=1)
    temp3 = np.concatenate((np.transpose(-C), np.zeros((m, p + m)), np.identity(m)), axis=1)
    temp4 = np.concatenate((np.zeros((m, n + p)), S, Lambda), axis=1)
    M = np.concatenate((temp1, temp2, temp3, temp4))
    
    return M, S, Lambda

def c5(A, G, C, g, x, gamma, lamb, s, bm, d):
    
    comp1 = G.dot(x) + g - A.dot(gamma) - C.dot(lamb)
    comp2 = bm - np.transpose(A).dot(x)
    comp3 = s + d - np.transpose(C).dot(x)
    comp4 = s * lamb
    
    return np.concatenate((comp1, comp2, comp3, comp4))

def solve_problem_c5(data_folder, max_iter=100, epsilon=1e-16, print_time=True, print_results=True):
    np.random.seed(42)
    random.seed(42)
    
    n = int(np.loadtxt(data_folder + "/G.dad")[:, 0][-1])
    p = n // 2
    m = 2 * n
    A = read_matrix(data_folder + "/A.dad", (n, p))
    bm = read_vector(data_folder + "/b.dad", p)
    C = read_matrix(data_folder + "/C.dad", (n, m))
    d = read_vector(data_folder + "/d.dad", m)
    e = np.ones(m)
    G = read_matrix(data_folder + "/g.dad", (n, n), True)
    g = np.zeros(n)
    x, gamma, lamb, s = initialize_variables(n, p)
    z = np.concatenate((x, gamma, lamb, s))

    if print_time:
        start_time = time.time()

    Mat, S, Lambda = M_KKT_C5(G, C, A, n, m, p, lamb, s)

    for i in range(max_iter):
        b = -c5(A, G, C, g, x, gamma, lamb, s, bm, d)
        delta = np.linalg.solve(Mat, b)
        alpha = calculate_step_size(lamb, delta[n + p:n + p + m], s, delta[n + m + p:])
        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[n + m + p:]).dot(lamb + alpha * delta[n + p:n + m + p])) / m
        sigma = (Mu / mu) ** 3
        b[n + m + p:] = b[n + p + m:] - np.diag(delta[n + p + m:] * delta[n + p:n + p + m]).dot(e) + sigma * mu * e
        delta = np.linalg.solve(Mat, b)
        alpha = calculate_step_size(lamb, delta[n + p:n + p + m], s, delta[n + m + p:])
        z = z + 0.95 * alpha * delta
        
        if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:n + m]) < epsilon) or (np.linalg.norm(-b[n + p:n + p + m]) < epsilon) or (np.abs(mu) < epsilon):
            break
        
        x, gamma, lamb, s = z[:n], z[n:n + p], z[n + p:n + m + p], z[n + m + p:]
        Mat, S, Lambda = M_KKT_C5(G, C, A, n, m, p, lamb, s)

    condition_number = np.linalg.cond(Mat)

    if print_time:
        end_time = time.time()
        print("Computation time: ", end_time - start_time)

    if print_results:
        print('Minimum found:', objective_function(x, G, g))
        print('Condition number:', condition_number)
        print('Iterations needed:', i)

print("optpr1 Results:\n")
solve_problem_c5(data_folder=r"optpr1", max_iter=100, epsilon=1e-16, print_time=True, print_results=True)
print("\n")
print("optpr2 Results:\n")
solve_problem_c5(data_folder=r"optpr2", max_iter=100, epsilon=1e-16, print_time=True, print_results=True)

def M_KKT_C6(G, C, A, n, m, p, lamb,s):
    
    S = np.diag(s)
    Lambda = np.diag(lamb)
    temp1 = np.concatenate((G,- A, - C),axis = 1)
    temp2 = np.concatenate((- np.transpose(A), np.zeros((p, p + m))), axis = 1)
    temp3 = np.concatenate((- np.transpose(C), np.zeros((m, p)), np.diag(-1 / lamb * s)), axis = 1)
    M = np.concatenate((temp1, temp2, temp3))
    
    return M, S, Lambda

def solve_ldlt(data_folder, max_iter=100, epsilon=1e-16, print_time=True, print_results=True):
    np.random.seed(42)
    random.seed(42)

    n = int(np.loadtxt(data_folder + "/G.dad")[:, 0][-1])
    p = n // 2
    m = 2 * n
    A = read_matrix(data_folder + "/A.dad", (n, p))
    bm = read_vector(data_folder + "/b.dad", p)
    C = read_matrix(data_folder + "/C.dad", (n, m))
    d = read_vector(data_folder + "/d.dad", m)
    e = np.ones(m)
    G = read_matrix(data_folder + "/G.dad", (n, n), True)
    g = np.zeros(n)
    x, gamma, lamb, s = initialize_variables(n, p)
    z = np.concatenate((x, gamma, lamb, s))

    if print_time:
        Start = time.time()

    Mat, S, Lamb = M_KKT_C6(G, C, A, n, m, p, lamb, s)

    for i in range(max_iter):
       lamb_inv = np.diag(1/lamb)
       b = c5(A, G, C, g, x, gamma, lamb, s, bm, d)
       r1, r2, r3, r4 = b[:n], b[n:n+p], b[n+p:n+p+m], b[n+p+m:]
       b = np.concatenate(([-r1,-r2,-r3+1/lamb*r4]))
       L, D, perm = ldl(Mat)
       y = np.linalg.solve(L, b)
       delta = np.linalg.solve(D.dot(np.transpose(L)), y)
       deltaS = lamb_inv.dot(- r4 - s * delta[(n + p):])
       delta = np.concatenate((delta, deltaS))
       alpha = calculate_step_size(lamb, delta[(n + p):(n + p + m)], s, delta[(n + m + p):])
       mu = s.dot(lamb) / m
       Mu = ((s + alpha * delta[(n + m + p):]).dot(lamb + alpha * delta[(n + p):(n + m + p)])) / m
       sigma = (Mu / mu) ** 3
       Ds = np.diag(delta[(n + p + m):] * delta[(n + p):(n + p + m)])
       b = np.concatenate((-r1, -r2, -r3 + lamb_inv.dot(r4 + Ds.dot(e) - sigma * mu * e)))
       y = np.linalg.solve(L, b)
       delta = np.linalg.solve(D.dot(np.transpose(L)), y)
       deltaS = lamb_inv.dot(- r4 - Ds.dot(e) + sigma * mu * e - s * delta[(n + p):])
       delta = np.concatenate((delta, deltaS))
       alpha = calculate_step_size(lamb, delta[(n + p):(n + p + m)], s, delta[(n + m + p):])
       z = z + 0.95 * alpha * delta

       if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:(n + m)]) < epsilon) or (np.linalg.norm(-b[(n + p):(n + p + m)]) < epsilon) or (np.abs(mu) < epsilon):
           break

       x, gamma, lamb, s = z[:n], z[n:(n + p)], z[(n + p):(n + m + p)], z[(n + m + p):]
       Mat, Lamb, S = M_KKT_C6(G, C, A, n, m, p, lamb, s)

    condition_number = np.linalg.cond(Mat)

    if print_time:
        End = time.time()
        print("Computation time: ", End - Start)

    if print_results:
        print('Minimum was found:', objective_function(x, G, g))
        print('Condition number:', condition_number)
        print('Iterations needed:', i)

print("\n")
print("optpr1 LDL^T Results:")
solve_ldlt(data_folder=r"optpr1", max_iter=100, epsilon=1e-16, print_time=True, print_results=True)
print("\n")
print("optpr2 LDL^T Results:")
solve_ldlt(data_folder=r"optpr2", max_iter=100, epsilon=1e-16, print_time=True, print_results=True)
