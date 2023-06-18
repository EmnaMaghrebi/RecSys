import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

def objective_function_U(U, R, V, sigma_u, sigma_v, sigma, d_dims, n_users):
    """
    Define the objective function to optimize with respect to U.

    Parameters:
    U: Matrix U to optimize.
    R: Matrix R.
    V: Matrix V.
    sigma_u: Standard deviation of the Gaussian distribution for U.
    sigma_v: Standard deviation of the Gaussian distribution for V.
    sigma: Standard deviation of the Gaussian distribution for noise.
    d_dims: Number of dimensions.
    n_users: Number of users.
    """
    U = U.reshape((d_dims, n_users))
    lambda_U = sigma / sigma_u
    lambda_V = sigma / sigma_v
    error = R - np.dot(U.T, V)
    regularization_term = lambda_U / 2 * np.linalg.norm(U)**2 + lambda_V / 2 * np.linalg.norm(V)**2
    return np.linalg.norm(error)**2 + regularization_term

def objective_function_V(V, R, U, sigma_u, sigma_v, sigma, d_dims, n_items):
    """
    Define the objective function to optimize with respect to V.

    Parameters:
    V: Matrix V to optimize.
    R: Matrix R.
    U: Matrix U.
    sigma_u: Standard deviation of the Gaussian distribution for U.
    sigma_v: Standard deviation of the Gaussian distribution for V.
    sigma: Standard deviation of the Gaussian distribution for noise.
    d_dims: Number of dimensions.
    n_items: Number of movies.
    """
    V = V.reshape((d_dims, n_items))
    lambda_U = sigma / sigma_u
    lambda_V = sigma / sigma_v
    error = R - np.dot(U.T, V)
    regularization_term = lambda_U / 2 * np.linalg.norm(U)**2 + lambda_V / 2 * np.linalg.norm(V)**2
    return np.linalg.norm(error)**2 + regularization_term

def frob(A, B):
    """
    Compute the Frobenius norm between two matrices.

    Parameters:
    A: First matrix.
    B: Second matrix.
    """
    return np.linalg.norm(A - B, 'fro')

def procrustes_norm(A, B, N):
    """
    Compute the Procrustes norm between two matrices.

    Parameters:
    A: First matrix.
    B: Second matrix.
    N: Length of matrices A and B.
    """
    _, _, disparity = procrustes(A, B)
    return disparity / np.sqrt(N)


def orth_procrustes(X, Y, len_X):
    """
    Compute the Procrustes solution (the optimal rotation and scaling) to minimize the 
    difference between two matrices, and return the root mean square deviation.

    Parameters:
    X: The first matrix.
    Y: The second matrix.
    len_X: The length (number of elements).

    Returns:
    The root mean square deviation between the Procrustes-transformed X and Y.
    """
    # Compute the Procrustes solution
    R, scale = orthogonal_procrustes(X, Y)

    # Compute the root mean square deviation
    rss = np.linalg.norm(X @ R - Y, 'fro') / np.sqrt(len_X)

    return rss
