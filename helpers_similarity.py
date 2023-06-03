import scipy.stats as stats
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# HELPER FUNCTION FOR TASK 1 AND 2
# REWRITE THE COMMENTS 
""""
"""

#generate matrices U, V and R from Gaussisan distributions
# mu: mean of the Gauss. distribution
# sigma_u: standard deviation of the Gauss. distribution for U
# sigma_v: standard deviation of the Gauss. distribution for V
# sigma: standard deviation of the Gauss. distribution for noise
# alpha: number of latent features

def generate_U_V_R(mu, sigma_u, sigma_v, sigma, d_dim, n_users, n_movies):
    """"
    """

    U = np.random.normal(mu, sigma_u, size = (d_dim,n_users))
    V = np.random.normal(mu, sigma_v, size = (d_dim,n_movies))
    noise = np.random.normal(mu, sigma, size = (n_users,n_movies)) 
    R = np.matmul(U.T,V) +noise
    return U, V, R

#map values to discrete ratings 
# x: value to be mapped

def condition(x):
    if x<-2:
        return 1
    elif x>=-2 and x<=-0.5:
        return 2
    elif x>-0.5 and x<=0.5:
        return 3
    elif x>0.5 and x<=2:
        return 4
    else:
        return 5

#compute similarity list of matrix U for a given user i
# U: matrix U
# user_i: user index

def similarity_list(A, user_i): 
    df_A = pd.DataFrame(A)
    sim_array= cosine_similarity(df_A)[user_i] 
    df_A['similarity_u1']= sim_array.tolist()
    df_A= df_A.sort_values(by= 'similarity_u1', ascending= False)
    user1_sim_list= df_A.index
    return user1_sim_list, sim_array

#compute kendalltau ranking similarity of 2 ranking lists from matrices A and B for k_best rankings
# A_list: list of users sorted by similarity to user 1 in matrix A
# B_list: list of users sorted by similarity to user 1 in matrix B
# k_best: number of best rankings to be considered

def ranking_similarity(A_list, B_list, k_best):
    tau, p_value = stats.kendalltau(A_list[:k_best], B_list[:k_best])
    return tau

#compute average of kendalltau ranking similarity for all users for given matrices A and B
# A: matrix A
# B: matrix B
# n_users: number of users
# k_best: number of best rankings to be considered

def sim_avg_users(A, B, n_users, k_best):
    sum_similarities= 0

    for user_i in range(n_users):
        _, similarity_list_A = similarity_list(A, user_i)
        _, similarity_list_B = similarity_list(B, user_i)
        kd_tau = ranking_similarity(similarity_list_A, similarity_list_B, k_best)
        sum_similarities+= kd_tau
        
    return sum_similarities/n_users    
        