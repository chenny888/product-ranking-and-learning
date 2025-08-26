import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression as LinearModel
import copy
import math
import pandas as pd

def AssortOpt(r, p, m):
    n = len(r)

    S = [[[] for i in range(n)] for j in range(m)];
    H = [[0 for i in range(n)] for j in range(m)]

    H[0][n - 1] = p[n - 1] * r[n - 1];
    S[0][n - 1] = [n - 1];

    sigma = {}

    for i in range(n - 2, -1, -1):
        H[0][i] = max(p[i] * r[i], H[0][i + 1]);

        if p[i] * r[i] > H[0][i + 1]:
            S[0][i] = [i];
        else:
            S[0][i] = S[0][i + 1];

        sigma[0] = (H[0][0], S[0][0])

    for k in range(1, m):

        for i in range(n - k - 1, -1, -1):

            if H[k - 1][i + 1] + p[i] * (r[i] - H[k - 1][i + 1]) > H[k][i + 1]:
                S[k][i] = copy.deepcopy(S[k - 1][i + 1]);
                S[k][i].append(i);
                H[k][i] = H[k - 1][i + 1] + p[i] * (r[i] - H[k - 1][i + 1]);
            else:
                S[k][i] = copy.deepcopy(S[k][i + 1]);
                H[k][i] = copy.deepcopy(H[k][i + 1]);

        sigma[k] = (H[k][0], S[k][0])

    return sigma


def ApproxOpt(r, p, G):
    M = len(G)

    sigma = AssortOpt(r, p, M)

    max_idx = 0

    lower_bound = 0

    H = []

    for key, values in sigma.items():

        R = values[0]
        H.append(R)
        if R * G[key] >= lower_bound:
            max_idx = key

            lower_bound = R * G[key]

    result = sigma[max_idx][1]
    result.reverse()

    return result, lower_bound, H, sigma[M - 1]


def UpperBound(R, G):
    M = len(G)

    result = 0
    for i in range(M - 1):
        result += R[i] * (G[i] - G[i + 1])

    result += R[M - 1] * G[M - 1]

    return result


def TrueReward(S, r, p, G):
    M = min(len(S), len(G))

    c = 1

    result = 0
    for i in range(M):
        result += c * p[i] * r[i] * G[i]

        c *= (1 - p[i])

    return result

from scipy.stats import bernoulli
from sklearn.preprocessing import normalize
import random


def featureGen(d, n, mu=0.25, sigma=1, normalized=True):
    x = np.random.normal(mu, sigma, (d, n))
    if normalized:
        x = normalize(x, axis=0, norm='l2')
    return x


def simulator(S, r, p, G):
    fx = []
    y = []

    result = 0

    u = np.random.uniform()
    X_span = len(G)
    for i in range(1, len(G)):
        if G[i] <= u:
            X_span = i
            break

    n = min(X_span, len(S))
    for i in range(n):
        rv = bernoulli.rvs(p[i])
        fx.append(S[i])
        y.append(rv)
        if rv:
            break

    return X_span, fx, np.array(y)