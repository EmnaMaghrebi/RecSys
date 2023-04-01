# generate imports for helper functions for similarity based methods  
import scipy.stats as stats
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#generate matrices U, V and R from Gauss. distributions
# mu: mean of the Gauss. distribution
# sigma_u: standard deviation of the Gauss. distribution for U
# sigma_v: standard deviation of the Gauss. distribution for V
# sigma: standard deviation of the Gauss. distribution for noise
# alpha: number of latent features

def generate_U_V_R(mu, sigma_u, sigma_v, sigma, alpha):
    U = np.random.normal(mu, sigma_u, size=(D,N))
    V = np.random.normal(mu, sigma_v, size=(D,M*alpha))
    noise = np.random.normal(mu, sigma, size=(N,M*alpha)) 
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

def similarity_list(U, user_i): 
    df_U = pd.DataFrame(U)
    sim_array= cosine_similarity(df_U)[user_i] 
    df_U['similarity_u1']= sim_array.tolist()
    df_U= df_U.sort_values(by= 'similarity_u1', ascending= False)
    user1_sim_list= df_U.index
    return user1_sim_list, sim_array

#compute kendalltau ranking similarity of 2 ranking lists for k_best rankings
# u1_list_U: list of users sorted by similarity to user 1 in matrix U
# u1_list_V: list of users sorted by similarity to user 1 in matrix V
# k_best: number of best rankings to be considered

def ranking_similarity(u1_list_U, u1_list_V, k_best):
    tau, p_value = stats.kendalltau(u1_list_U[:k_best], u1_list_V[:k_best])
    return tau

#compute average of kendalltau similarity for all users for given matrices U and R

def sim_avg_users(A, B):
    sum_similarities= 0
    for user_i in range(N):
        _, similarity_list_A = similarity_list(A, user_i)
        _, similarity_list_B = similarity_list(B, user_i)
        # print("user: ", user_i, "\n similarity U:\n", similarity_list_U, "\n similarity U_svd:\n", similarity_list_R)
        kd_tau = ranking_similarity(similarity_list_A, similarity_list_B, k_best)
        # print("kendall tau:", kd_tau)
        sum_similarities+= kd_tau
    return sum_similarities/N    
        