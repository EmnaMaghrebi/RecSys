import numpy as np
from scipy.optimize import minimize
from helpers_optimization import *
from helpers_similarity import *

# set up the dimensions
D = 2
#M = 10
N_vals = [10, 20, 30, 40, 50 ]
#N = 10
mu = 0.0
M_vals = [10, 20, 30, 40, 50 ]
# set up the parameters
sigma = 0.5
sigma_u = sigma_v = 1

# generate the matrices R and U from a Gaussian distribution
#R = np.random.randn(N, M)
#U = np.random.randn(D, N)
np.random.seed(43)


# define the objective function
def objective_function_V(V, R, U, sigma,M):
    V = V.reshape((D, M))  # reshape V from a vector to a matrix
    lambda_U = sigma / sigma_u
    lambda_V = sigma / sigma_v
    error = R - np.dot(U.T, V)
    regularization_term = lambda_U / 2 * np.linalg.norm(U) ** 2 + lambda_V / 2 * np.linalg.norm(V) ** 2
    return np.sum(error ** 2) * 1 / 2 + regularization_term

def objective_function_U(U, R, V, sigma,N):
    U = U.reshape((D, N))  # reshape V from a vector to a matrix
    lambda_U = sigma / sigma_u
    lambda_V = sigma / sigma_v
    error = R - np.dot(U.T, V)
    regularization_term = lambda_U / 2 * np.linalg.norm(U) ** 2 + lambda_V / 2 * np.linalg.norm(U) ** 2
    return np.sum(error ** 2) * 1 / 2 + regularization_term

def result(N,M):

    U, V, R = generate_U_V_R(mu, sigma_u, sigma_v, sigma, D, N, M)
    # set up the initial values for V
    V0 = np.random.randn(D * M)

    # set up the constraints
    #constraints = ({'type': 'eq', 'fun': lambda V: np.linalg.norm(V) - 1})

    # minimize the objective function
    result = minimize(objective_function_V, V0, args=(R, U, sigma,M))
    if result.success :
        V_result = result.x.reshape((D, M))
    else :
        print('Minimization failure for U')
    V_result = result.x.reshape((D, M))
    # print the optimized value of V
    diff_norm_V = frob(V, V_result)
    print(f'diff norm v : {diff_norm_V}' )

for j in range (len(M_vals) ):
    for i in range(len(N_vals)):
        n = N_vals[i]
        m = M_vals[j]
        print(f'N = {n} , M = {m}')
        result(n,m)