import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats


# define the objective function to optimize wrt to U
# U: matrix U to optimize
# R: matrix R
# V: matrix V
# sigma_u: standard deviation of the Gauss. distribution for U
# sigma_v: standard deviation of the Gauss. distribution for V
# sigma: standard deviation of the Gauss. distribution for noise
# d_dims: number of dimensions
# n_users: number of users

def objective_function_U(U, R, V, sigma_u, sigma_v, sigma, d_dims, n_users):
    U = U.reshape((d_dims, n_users))
    lambda_U = sigma/sigma_u
    lambda_V = sigma/sigma_v
    error = R - np.dot(U.T, V)
    regularization_term = lambda_U/2 * np.linalg.norm(U)**2 + lambda_V/2 * np.linalg.norm(V)**2
    return  np.sum(error**2)*1/2 + regularization_term


# define the objective function to optimize wrt to V
# V: matrix V to optimize
# R: matrix R
# U: matrix U
# sigma_u: standard deviation of the Gauss. distribution for U
# sigma_v: standard deviation of the Gauss. distribution for V
# sigma: standard deviation of the Gauss. distribution for noise
# d_dims: number of dimensions
# n_movies: number of movies
# alpha: scale factor for M

def objective_function_V(V, R, U, sigma_u, sigma_v, sigma, d_dims, n_movies):
    V = V.reshape((d_dims, n_movies))
    lambda_U = sigma/sigma_u
    lambda_V = sigma/sigma_v
    error = R - np.dot(U.T, V)
    regularization_term = lambda_U/2 * np.linalg.norm(U)**2 + lambda_V/2 * np.linalg.norm(V)**2
    return  np.sum(error**2)*1/2 + regularization_term


#compute scaled Frobenius norm to be independent of the size of the compared matrices 
#explain the parameters of the function 
# A: matrix to be compared
# B: matrix to be compared
# N: length of matrix A and B
# return: scaled Frobenius norm  

def frob(A, B, M):
    return np.linalg.norm(A - B, 'fro')/np.sqrt(M)


#Procrustes norm
# A: matrix to be compared
# B: matrix to be compared
# N: length of matrix A and B
# return: scaled Procrustes norm

def procrustes_norm(A, B, N):
    _, _, disparity = procrustes(A, B)
    return disparity/np.sqrt(N) 

