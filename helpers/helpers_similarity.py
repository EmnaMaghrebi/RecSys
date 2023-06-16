import scipy.stats as stats
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_U_V_R(mu, sigma_u, sigma_v, sigma, d_dim, n_users, n_items):
    """
    Generate matrices U, V and R from Gaussian distributions.

    Parameters:
    mu: Mean of the Gaussian distribution.
    sigma_u: Standard deviation of the Gaussian distribution for U.
    sigma_v: Standard deviation of the Gaussian distribution for V.
    sigma: Standard deviation of the Gaussian distribution for noise.
    d_dim: Dimensionality of the latent feature space.
    n_users: Number of users.
    n_items: Number of items.
    """
    U = np.random.normal(mu, sigma_u, size=(d_dim, n_users))
    V = np.random.normal(mu, sigma_v, size=(d_dim, n_items))
    noise = np.random.normal(mu, sigma, size=(n_users, n_items))
    R = np.matmul(U.T, V) + noise
    return U, V, R

def condition(x):
    """
    Change values to discrete ratings.

    Parameters:
    x: Value to be changed.
    """
    if x < -2:
        return 1
    elif -2 <= x <= -0.5:
        return 2
    elif -0.5 < x <= 0.5:
        return 3
    elif 0.5 < x <= 2:
        return 4
    else:
        return 5

def similarity_list(A, user):
    """
    Compute similarity list of a matrix A for a given user .

    Parameters:
    A: Matrix A.
    user: User index.
    """
    df_A = pd.DataFrame(A)
    sim_array = cosine_similarity(df_A)[user]
    df_A['similarity_u1'] = sim_array.tolist()
    df_A = df_A.sort_values(by='similarity_u1', ascending=False)
    user1_sim_list = df_A.index
    return user1_sim_list, sim_array

def ranking_similarity(list_A, list_B, k_best):
    """
    Compute Kendall Tau ranking similarity of 2 ranking lists from matrices A and B for k_best rankings.

    Parameters:
    A_list: List of users sorted by similarity to user 1 in matrix A.
    B_list: List of users sorted by similarity to user 1 in matrix B.
    k_best: Number of best rankings to be considered.
    """
    tau, _ = stats.kendalltau(list_A[:k_best], list_B[:k_best])
    return tau

def sim_avg_users(A, B, n_users, k_best):
    """
    Compute average of Kendall Tau ranking similarity for all users for given matrices A and B.

    Parameters:
    A: Matrix A.
    B: Matrix B.
    n_users: Number of users.
    k_best: Number of best rankings to be considered.
    """
    sum_similarities = 0
    for user in range(n_users):
        _, similarity_list_A = similarity_list(A, user)
        _, similarity_list_B = similarity_list(B, user)
        kd_tau = ranking_similarity(similarity_list_A, similarity_list_B, k_best)
        sum_similarities += kd_tau
    return sum_similarities / n_users
