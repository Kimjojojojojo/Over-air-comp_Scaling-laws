import numpy as np


def OAC(K, P, sigma, h):
    g = np.zeros(K)
    for k in range(K):
        tmp1 = np.sqrt(P) * np.sum([h[i] for i in range(k+1)])
        tmp2 = sigma + P * np.sum([h[i]**2 for i in range(k+1)])
        g[k] = tmp1 / tmp2

    i_star = np.argmax(g)

    S = np.zeros(K+1)
    for k in range(K):
        S[k] = 1 / (h[k] * np.sqrt(P))
    S[K] = 0

    a = np.zeros(K) # post-processing
    for k in range(K):
        if g[k] <= S[k + 1]:
            a[k] = S[k + 1]
        if g[k] > S[k]:
            a[k] = S[k]
        if g[k] > S[k + 1] and g[k] <= S[k]:
            a[k] = g[k]
    a_star = a[i_star] # optimal a

    b = np.zeros(K) # pre-processing of each user
    for k in range(i_star+1):
        b[k] = np.sqrt(P)
    for k in range(i_star+1, K):
        b[k] = 1 / (a[i_star] * h[k])

    c = np.zeros(K) # ch inverse pre-processing
    for k in range(K):
        c[k] = 1 / (a[i_star] * h[k])

    MSE = np.zeros(K) # MSE of sum
    for i in range(K):
        tmp1 = np.sum([ (a[i] * h[k] * np.sqrt(P) - 1)**2 for k in range(i+1) ])
        tmp2 = sigma * a[i]**2
        MSE[i] = tmp1 + tmp2

    PW = np.sum(np.abs(b)**2) # Power

    return MSE, PW, i_star

def OAC_CH_inversion(K, P, sigma, h): # i_star = 1
    b = np.zeros(K)
    a = 0
    for k in range(K):
        b[k] = np.sqrt(P) * (h[1] / h[k])
    a = 1 / (np.sqrt(P) * h[1])

    MSE_OAC = np.zeros(K)  # MSE of sum
    for i in range(K):
        tmp1 = np.sum([(a[i] * h[k] * np.sqrt(P) - 1) ** 2 for k in range(i + 1)])
        tmp2 = sigma * a[i] ** 2
        MSE_OAC[i] = tmp1 + tmp2

    PW_OAC = np.sum(np.abs(b) ** 2)  # Power
