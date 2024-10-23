import numpy as np
import matplotlib.pyplot as plt
import OAC as oac

# 초기 설정
K = 10
sigma = 1  # sigma^2
num_samples = 1000  # 각 P마다 100만 개의 샘플 생성

# dB 단위에서의 P 설정 (0dB, 5dB, 10dB, 15dB, 20dB)
P_dB = np.arange(0, 25, 2.5)
P_values = 10 ** (P_dB / 10)  # dB 값을 선형 값으로 변환

K_range = np.arange(5,101,5)

# 결과 저장용 리스트
#### results list ####
# OAC policy
MSE_OAC_list = []
# CH inversion policy
MSE_OAC_CH_inversion_list = []
# Energy greedy policy
MSE_OAC_Energy_greedy_list = []
# First i=1 and i=2 policy
MSE_OAC_first_i_1_list = []
MSE_OAC_first_i_2_list = []

# P 값을 변화시키며 MSE_OAC와 PW_OAC 계산
for kk in K_range:
    P = 10
    #### samples list ####
    # OAC policy
    MSE_OAC_samples = []
    # CH inversion policy
    MSE_OAC_CH_inversion_samples = []
    # Energy greedy policy
    MSE_OAC_Energy_greedy_samples = []
    # First i=1 and i=2 policy
    MSE_OAC_first_i_1_samples = []
    MSE_OAC_first_i_2_samples = []


    # 각 P에 대해 num_samples 개의 샘플 생성
    for _ in range(num_samples):
        # h 생성 및 정렬
        h = np.abs(np.random.normal(0, 1, kk))
        #h = (np.random.normal(0, 1, kk) + 1j*np.random.normal(0, 1, kk))/np.sqrt(2)
        h_ordered = np.sort(h)

        #### MSE, PW, ESR calculation ####
        # OAC policy
        MSE_OAC, _, i_star = oac.OAC(kk, P, sigma, h_ordered)
        # CH inversion policy
        MSE_OAC_CH_inversion, _ = oac.OAC_CH_inversion(kk, P, sigma, h_ordered)
        # Energy greedy policy (OAC_Energy_greedy)
        MSE_OAC_Energy_greedy, _ = oac.OAC_Energy_greedy(kk, P, sigma, h_ordered)
        # First i#1 policy
        MSE_OAC_first_i_1, _ = oac.first_i(kk, P, sigma, h_ordered, idx = 1)
        # First i#2 policy
        MSE_OAC_first_i_2, _ = oac.first_i(kk, P, sigma, h_ordered, idx = 2)

        #### samples append ####
        # OAC policy
        MSE_OAC_samples.append(MSE_OAC[i_star])
        # CH inversion policy
        MSE_OAC_CH_inversion_samples.append(MSE_OAC_CH_inversion)
        # Energy greedy policy
        MSE_OAC_Energy_greedy_samples.append(MSE_OAC_Energy_greedy)
        # First i=1 and i=2 policy
        MSE_OAC_first_i_1_samples.append(MSE_OAC_first_i_1)
        MSE_OAC_first_i_2_samples.append(MSE_OAC_first_i_2)

    #### append to list ####
    # OAC policy
    MSE_OAC_list.append(np.mean(MSE_OAC_samples))
    # CH inversion policy
    MSE_OAC_CH_inversion_list.append(np.mean(MSE_OAC_CH_inversion_samples))
    # Energy greedy policy
    MSE_OAC_Energy_greedy_list.append(np.mean(MSE_OAC_Energy_greedy_samples))
    # First i=1 and i=2 policy
    MSE_OAC_first_i_1_list.append(np.mean(MSE_OAC_first_i_1_samples))
    MSE_OAC_first_i_2_list.append(np.mean(MSE_OAC_first_i_2_samples))

# 그래프 출력 (MSE_OAC vs P(dB))
plt.figure(figsize=(10, 5))
plt.semilogy(K_range, MSE_OAC_list, marker='o', linestyle='-', color='b', label='OAC')
# plt.semilogy(K_range, MSE_OAC_CH_inversion_list, marker='v', linestyle='-', color='g', label='CH Inversion')
plt.semilogy(K_range, MSE_OAC_Energy_greedy_list, marker='s', linestyle='-', color='green', label='Energy Greedy')
plt.semilogy(K_range, MSE_OAC_first_i_1_list, marker='x', linestyle='-', color='r', label='i = $\sqrt{K}$')
plt.semilogy(K_range, MSE_OAC_first_i_2_list, marker='+', linestyle='-', color='black', label='i = $K / 2$')
plt.title('Average MSE_OAC vs user (in dB)')
plt.xlabel('user (dB)')
plt.ylabel('Average MSE_OAC')
plt.ylim([10**(-2), 1])
plt.legend()
plt.grid(True)
plt.show()

# ESR_OAC vs P(dB) 그래프 출력
# plt.figure(figsize=(10, 5))
# plt.plot(K_range, ESR_OAC_list, marker='o', linestyle='-', color='r', label='OAC')
# plt.plot(K_range, ESR_OAC_CH_inversion_list, marker='v', linestyle='-', color='g', label='CH Inversion')
# plt.plot(K_range, ESR_OAC_Energy_greedy_list, marker='s', linestyle='-', color='b', label='Energy Greedy')
# plt.plot(K_range, ESR_OAC_first_i_1_list, marker='x', linestyle='-', color='purple', label='i = $\sqrt{K}$')
# plt.plot(K_range, ESR_OAC_first_i_2_list, marker='+', linestyle='-', color='cyan', label='i = $K / 2$')
# plt.title('Average ESR_OAC vs user (in dB)')
# plt.xlabel('user')
# plt.ylabel('Average ESR_OAC [%]')
# plt.ylim(0, 100)
# plt.legend()
# plt.grid(True)
# plt.show()
