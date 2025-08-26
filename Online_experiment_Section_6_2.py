import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression as LinearModel
import copy
import math
import pandas as pd
from scipy.stats import bernoulli
from sklearn.preprocessing import normalize
import random
import time
from utils import *



d = 10
N = 1000  # number of products
M = 20  # length of attention span
G = [0.95 ** i for i in range(M)]  # attention span
customer_d = 5
dd = customer_d * d  # 50
inv_e = 0.36787944
T = 10000  # num of customers in each simulation

random.seed(30)
np.random.seed(0)
p_threshold = 1
theta = np.random.normal(0.25, 1, (dd, 1))  # generate theta N(0.25, 1)
theta = 0.906 * normalize(theta, axis=0, norm='l2')

Iters = [i+1 for i in range(0,10)]   # Independent simulations

for iteration in Iters:

    random.seed(iteration)
    np.random.seed(iteration)
    print('Conduct the %d-th experiment.' % (iteration))
    data = {}
    Res_hist = np.zeros(T)
    KW_Res_hist = np.zeros(T)

    h_LB_err = {}
    h_est_err = {}
    for i in range(M - 1):
        h_LB_err[i] = np.zeros(T)
        h_est_err[i] = np.zeros(T)

    import time

    start = time.perf_counter()

    ### Online Learning
    Res = 0
    KW_Res = 0

    X = []
    Y = []

    h = 0.05 * np.ones(M-1)

    G_u = np.ones(dd)

    theta_err = np.zeros(T)

    N_t = np.ones(M)
    N_c = np.zeros(M)
    est_theta = np.zeros(dd)

    prod_feat = featureGen(d, N, normalized=False)  # generate prod feature N(0.25, 1) with d=10 for N=1000 products

    V = np.eye(dd)
    V_inv = np.eye(dd)
    XtY = np.zeros(dd)

    # randomly generate price r
    r = np.random.rand(N) * 10
    r_sorted = np.sort(r)[::-1]
    r = r_sorted


    for t in range(T):

        rho = 1
        cust_feat = featureGen(customer_d, 1, mu=1, sigma=0.1, normalized=True)  # N(1, 0.1) with dim=5, normalize to 1,  |\theta|^* <= 1

        if t % 100 == 0:
            import time

            end = time.perf_counter()
            time = end - start
            print(f"t = {t}, collapsed time = {time} seconds")


        feat = []

        feat = np.einsum("ik,jl->ijkl", prod_feat, cust_feat)
        feat = feat.reshape(-1, N)  ## N x dd

        # normalize feature if p >=1 or p<=0
        # norm_x = np.linalg.norm(feature, axis=0)
        # feature are set as norm =1,
        # *-1 feature if p<0
        p = np.dot(feat.T, theta[:, 0])
        feat_norm = np.linalg.norm(feat, axis=0)
        idx_p = feat_norm >= p_threshold
        feat[:, idx_p] = p_threshold* normalize(feat[:, idx_p], axis=0, norm="l2")
        p[idx_p] = p_threshold * p[idx_p] / feat_norm[idx_p]

        idx_p = p < 0
        feat[:, idx_p] = - feat[:, idx_p]
        p[idx_p] = -p[idx_p]

        # update u
        u = np.dot(feat.T, est_theta)

        # temp = |x|_V_inv
        tmp = np.dot(feat.T, V_inv)
        tmp = np.einsum('ij,ji->i', tmp, feat)
        tmp = np.sqrt(tmp)
        tmp = np.array(tmp)
        u += rho * tmp
        u = np.clip(u, 0, 1)

        # obtain approximation solution under optimistic estimators, S solved by RankUCB
        S, _, H, Sigma = ApproxOpt(r, u, G_u)
        # greedily fill the empty slots
        s_plus = list(set(Sigma[1]) - set(S))
        s_plus.sort()
        S += s_plus

        res = TrueReward(S, r[S], p[S], G)  # true reward for S
        # simulator customer behavior with
        X_span, obs, y = simulator(S, r[S], p[S], G)

        X = feat[:, obs]
        XtY += X @ y
        XtX = np.einsum("ik,jk->ij", X, X)
        V += XtX

        V_inv = np.linalg.inv(V)

        est_theta = V_inv @ XtY

        # update true reward
        Res += res
        Res_hist[t] = Res

        # estimat h
        for k in range(len(y)):
            if y[k] == 0:  # h[k] is observed
                N_t[k] += 1
                # if k <= len(y) - 2:  # Not the last item
                #     N_c[k] += 1
        if y[-1] == 0 and len(y) < M:  # the last attempt was failure only if customer views the last product but didn't buy & it is not end of M
            N_c[len(y)-1] += 1
        # update optimistic estimator h_L
        # h_est = [1 - min(float(N_c[k] / N_t[k]), 1) for k in range(M - 1)]
        h_est = [float(N_c[k] / N_t[k]) for k in range(M - 1)]

        h_L = [h_est[k] - np.sqrt(np.log(t+1) / N_t[k]) for k in range(M - 1)]
        for k in range(M - 1):
            (h_LB_err[k])[t] += abs(h_L[k] - h[k])
            (h_est_err[k])[t] += abs(h_est[k] - h[k])

        # update h_L due to IFR
        if t % 100 ==0:
            print(f"N_t = {N_t}")
            print(f"h_est = {h_est}")
            print(f"h_L = {h_L}")

        h_L = np.clip(h_L, 0, 1)

        # update G_u
        G_u = np.ones(M)
        for s in range(1, M):
            G_u[s] = G_u[s - 1] * (1 - h_L[s - 1])

        #### calculate the reward by approxOpt with known choice parameters
        kw_S, _, H, _ = ApproxOpt(r, p, G)
        kw_res = TrueReward(kw_S, r[kw_S], p[kw_S], G)
        KW_Res += kw_res
        KW_Res_hist[t] = KW_Res

    data["h_LB_err"]  = h_LB_err
    data["h_est_err"] = h_est_err
    data["Res"] = Res_hist
    data["KW_Res"] = KW_Res_hist

    import pickle
    with open(f'online_data_iter_{iteration}.pickle', 'wb') as f: 
        pickle.dump(data, f)

