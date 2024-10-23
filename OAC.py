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

    b = np.zeros((K,K)) # pre-processing of each user
    for i in range(K):
        for k in range(i+1):
            b[i][k] = np.sqrt(P)
        for k in range(i+1, K):
            b[i][k] = 1 / (a[i] * h[k])

    for k in range(K):
        if b[i_star][k] > P:
            print(f'[OAC]no valid b:{b[i_star][k]}')

    MSE = np.zeros(K) # MSE of sum
    for i in range(K):
        tmp1 = np.sum([(a[i] * h[k] * b[i][k] - 1)**2 for k in range(K)])
        tmp2 = sigma * a[i]**2
        # tmp3 = np.sum([(a[i] * h[k] * b[i][k] - 1)**2 for k in range(K)])
        # tmp4 = sigma * a[i]**2
        MSE[i] = tmp1 + tmp2

    PW = np.zeros(K)
    for k in range(K):
        PW[k] = np.sum(np.abs(b[k])**2) # Power

    return MSE/K, PW, i_star

def OAC_CH_inversion(K, P, sigma, h): # i_star = 1
    b = np.zeros(K)
    a = 0
    for k in range(K):
        b[k] = np.sqrt(P) * (h[0] / h[k])
    a = 1 / (np.sqrt(P) * h[0])

    # tmp1 = np.sum([(a * h[k] * b[k] - 1) ** 2 for k in range(K)])
    # print(tmp1)
    tmp2 = sigma * (a ** 2)
    MSE =  tmp2
    #print(MSE, P, h)
    PW = np.sum(np.abs(b) ** 2)  # Power

    return MSE/K, PW


def OAC_Energy_greedy(K, P, sigma, h): # i_star = K
    b = np.full(K, np.sqrt(P))

    tmp1 = 1 / (np.sqrt(P) * h[-1])
    tmp2_1 = np.sum(h)
    tmp2_2 = np.sum(h**2)
    tmp2 = (np.sqrt(P) * tmp2_1) / (sigma + P * tmp2_2)

    a = min(tmp1, tmp2)

    tmp3 = np.sum([(a * h[k] * b[k] - 1) ** 2 for k in range(K)])
    tmp4 = sigma * (a ** 2)
    MSE = tmp3 + tmp4
    # print(MSE, P, h)
    PW = np.sum(np.abs(b) ** 2)  # Power

    return MSE/K, PW

def first_i(K, P, sigma, h, idx):
    i = 0
    if idx == 1:
        i = max(1, int(np.floor(np.sqrt(K))))
    if idx == 2:
        i = max(1, int(np.floor(K / 2)))

    a = (1/h[i] + 1/h[i-1])/(2*np.sqrt(P))

    b = np.zeros(K)  # pre-processing of each user
    for k in range(i):
        b[k] = np.sqrt(P)
    for k in range(i, K):
        b[k] = 1 / (a * h[k])

    for k in range(K):
        if b[k]**2 > P+0.1:
            print(f'[first_i{idx}]no valid b:{b[k]**2}')

    tmp1 = np.sum([(a * h[k] * b[k] - 1) ** 2 for k in range(K)])
    tmp2 = sigma * (a ** 2)
    MSE = tmp1 + tmp2
    # print(MSE, P, h)
    PW = np.sum(np.abs(b) ** 2)  # Power

    return MSE/K, PW
