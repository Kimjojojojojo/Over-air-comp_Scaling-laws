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

K_range = np.arange(10,100,1)

# 결과 저장용 리스트
#### results list ####
# OAC policy
MSE_OAC_list = []
PW_OAC_list = []
ESR_OAC_list = []  # energy savings rate
# CH inversion policy
MSE_OAC_CH_inversion_list = []
PW_OAC_CH_inversion_list = []
ESR_OAC_CH_inversion_list = []  # energy savings rate
# Energy greedy policy
MSE_OAC_Energy_greedy_list = []
PW_OAC_Energy_greedy_list = []
ESR_OAC_Energy_greedy_list = []  # energy savings rate
# First i=1 and i=2 policy
MSE_OAC_first_i_1_list = []
PW_OAC_first_i_1_list = []
ESR_OAC_first_i_1_list = []
MSE_OAC_first_i_2_list = []
PW_OAC_first_i_2_list = []
ESR_OAC_first_i_2_list = []

# P 값을 변화시키며 MSE_OAC와 PW_OAC 계산
for P in P_values:
    #### samples list ####
    # OAC policy
    MSE_OAC_samples = []
    PW_OAC_samples = []
    ESR_OAC_samples = []
    # CH inversion policy
    MSE_OAC_CH_inversion_samples = []
    PW_OAC_CH_inversion_samples = []
    ESR_OAC_CH_inversion_samples = []
    # Energy greedy policy
    MSE_OAC_Energy_greedy_samples = []
    PW_OAC_Energy_greedy_samples = []
    ESR_OAC_Energy_greedy_samples = []
    # First i=1 and i=2 policy
    MSE_OAC_first_i_1_samples = []
    PW_OAC_first_i_1_samples = []
    ESR_OAC_first_i_1_samples = []

    MSE_OAC_first_i_2_samples = []
    PW_OAC_first_i_2_samples = []
    ESR_OAC_first_i_2_samples = []

    # 각 P에 대해 num_samples 개의 샘플 생성
    for _ in range(num_samples):
        # h 생성 및 정렬
        h = np.abs(np.random.normal(0, 1, K))
        h_ordered = np.sort(h)

        #### MSE, PW, ESR calculation ####
        # OAC policy
        MSE_OAC, PW_OAC, i_star = oac.OAC(K, P, sigma, h_ordered)
        ESR_OAC = 100 - PW_OAC[i_star] / (K * P) * 100  # energy efficiency

        # CH inversion policy
        MSE_OAC_CH_inversion, PW_OAC_CH_inversion = oac.OAC_CH_inversion(K, P, sigma, h_ordered)
        ESR_OAC_CH_inversion = 100 - PW_OAC_CH_inversion / (K * P) * 100

        # Energy greedy policy (OAC_Energy_greedy)
        MSE_OAC_Energy_greedy, PW_OAC_Energy_greedy = oac.OAC_Energy_greedy(K, P, sigma, h_ordered)
        ESR_OAC_Energy_greedy = 100 - PW_OAC_Energy_greedy / (K * P) * 100

        # First i#1 policy
        i_sqrt_K = max(1, int(np.floor(np.sqrt(K))))
        MSE_OAC_first_i_1 = MSE_OAC[i_sqrt_K]
        PW_OAC_first_i_1 = PW_OAC[i_sqrt_K]
        ESR_OAC_first_i_1 = 100 - PW_OAC_first_i_1 / (K * P) * 100

        # First i#2 policy
        i_half_K = max(1, int(np.floor(K/2)))
        MSE_OAC_first_i_2 = MSE_OAC[i_half_K]
        PW_OAC_first_i_2 = PW_OAC[i_half_K]
        ESR_OAC_first_i_2 = 100 - PW_OAC_first_i_2 / (K * P) * 100

        #### samples append ####
        # OAC policy
        MSE_OAC_samples.append(MSE_OAC[i_star])
        PW_OAC_samples.append(PW_OAC)
        ESR_OAC_samples.append(ESR_OAC)

        # CH inversion policy
        MSE_OAC_CH_inversion_samples.append(MSE_OAC_CH_inversion)
        PW_OAC_CH_inversion_samples.append(PW_OAC_CH_inversion)
        ESR_OAC_CH_inversion_samples.append(ESR_OAC_CH_inversion)

        # Energy greedy policy
        MSE_OAC_Energy_greedy_samples.append(MSE_OAC_Energy_greedy)
        PW_OAC_Energy_greedy_samples.append(PW_OAC_Energy_greedy)
        ESR_OAC_Energy_greedy_samples.append(ESR_OAC_Energy_greedy)

        # First i=1 and i=2 policy
        MSE_OAC_first_i_1_samples.append(MSE_OAC_first_i_1)
        PW_OAC_first_i_1_samples.append(PW_OAC_first_i_1)
        ESR_OAC_first_i_1_samples.append(ESR_OAC_first_i_1)

        MSE_OAC_first_i_2_samples.append(MSE_OAC_first_i_2)
        PW_OAC_first_i_2_samples.append(PW_OAC_first_i_2)
        ESR_OAC_first_i_2_samples.append(ESR_OAC_first_i_2)

    #### samples average ####
    # OAC policy
    MSE_OAC_avg = np.mean(MSE_OAC_samples)
    PW_OAC_avg = np.mean(PW_OAC_samples)
    ESR_OAC_avg = np.mean(ESR_OAC_samples)

    # CH inversion policy
    MSE_OAC_CH_inversion_avg = np.mean(MSE_OAC_CH_inversion_samples)
    PW_OAC_CH_inversion_avg = np.mean(PW_OAC_CH_inversion_samples)
    ESR_OAC_CH_inversion_avg = np.mean(ESR_OAC_CH_inversion_samples)

    # Energy greedy policy
    MSE_OAC_Energy_greedy_avg = np.mean(MSE_OAC_Energy_greedy_samples)
    PW_OAC_Energy_greedy_avg = np.mean(PW_OAC_Energy_greedy_samples)
    ESR_OAC_Energy_greedy_avg = np.mean(ESR_OAC_Energy_greedy_samples)

    # First i=1 and i=2 policy
    MSE_OAC_first_i_1_avg = np.mean(MSE_OAC_first_i_1_samples)
    PW_OAC_first_i_1_avg = np.mean(PW_OAC_first_i_1_samples)
    ESR_OAC_first_i_1_avg = np.mean(ESR_OAC_first_i_1_samples)

    MSE_OAC_first_i_2_avg = np.mean(MSE_OAC_first_i_2_samples)
    PW_OAC_first_i_2_avg = np.mean(PW_OAC_first_i_2_samples)
    ESR_OAC_first_i_2_avg = np.mean(ESR_OAC_first_i_2_samples)

    #### append to list ####
    # OAC policy
    MSE_OAC_list.append(MSE_OAC_avg/K)
    PW_OAC_list.append(PW_OAC_avg/K)
    ESR_OAC_list.append(ESR_OAC_avg)

    # CH inversion policy
    MSE_OAC_CH_inversion_list.append(MSE_OAC_CH_inversion_avg/K)
    PW_OAC_CH_inversion_list.append(PW_OAC_CH_inversion_avg/K)
    ESR_OAC_CH_inversion_list.append(ESR_OAC_CH_inversion_avg)

    # Energy greedy policy
    MSE_OAC_Energy_greedy_list.append(MSE_OAC_Energy_greedy_avg/K)
    PW_OAC_Energy_greedy_list.append(PW_OAC_Energy_greedy_avg/K)
    ESR_OAC_Energy_greedy_list.append(ESR_OAC_Energy_greedy_avg)

    # First i=1 and i=2 policy
    MSE_OAC_first_i_1_list.append(MSE_OAC_first_i_1_avg/K)
    PW_OAC_first_i_1_list.append(PW_OAC_first_i_1_avg/K)
    ESR_OAC_first_i_1_list.append(ESR_OAC_first_i_1_avg)

    MSE_OAC_first_i_2_list.append(MSE_OAC_first_i_2_avg/K)
    PW_OAC_first_i_2_list.append(PW_OAC_first_i_2_avg/K)
    ESR_OAC_first_i_2_list.append(ESR_OAC_first_i_2_avg)

# 그래프 출력 (MSE_OAC vs P(dB))
plt.figure(figsize=(10, 5))
plt.plot(P_dB, MSE_OAC_list, marker='o', linestyle='-', color='r', label='OAC')
plt.plot(P_dB, MSE_OAC_CH_inversion_list, marker='v', linestyle='-', color='g', label='CH Inversion')
plt.plot(P_dB, MSE_OAC_Energy_greedy_list, marker='s', linestyle='-', color='b', label='Energy Greedy')
plt.plot(P_dB, MSE_OAC_first_i_1_list, marker='x', linestyle='-', color='purple', label='i = $\sqrt{K}$')
plt.plot(P_dB, MSE_OAC_first_i_2_list, marker='+', linestyle='-', color='cyan', label='i = $K / 2$')
plt.title('Average MSE_OAC vs P (in dB)')
plt.xlabel('P (dB)')
plt.ylabel('Average MSE_OAC')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# ESR_OAC vs P(dB) 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(P_dB, ESR_OAC_list, marker='o', linestyle='-', color='r', label='OAC')
plt.plot(P_dB, ESR_OAC_CH_inversion_list, marker='v', linestyle='-', color='g', label='CH Inversion')
plt.plot(P_dB, ESR_OAC_Energy_greedy_list, marker='s', linestyle='-', color='b', label='Energy Greedy')
plt.plot(P_dB, ESR_OAC_first_i_1_list, marker='x', linestyle='-', color='purple', label='i = $\sqrt{K}$')
plt.plot(P_dB, ESR_OAC_first_i_2_list, marker='+', linestyle='-', color='cyan', label='i = $K / 2$')
plt.title('Average ESR_OAC vs P (in dB)')
plt.xlabel('P (dB)')
plt.ylabel('Average ESR_OAC [%]')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()
