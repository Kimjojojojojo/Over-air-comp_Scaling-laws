import numpy as np
import matplotlib.pyplot as plt
import OAC as oac

# 초기 설정
K = 10
sigma = 1  # sigma^2
num_samples = 100000  # 각 P마다 100만 개의 샘플 생성

# dB 단위에서의 P 설정 (0dB, 5dB, 10dB, 15dB, 20dB)
P_dB = np.arange(0, 25, 2.5)
P_values = 10 ** (P_dB / 10)  # dB 값을 선형 값으로 변환

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

    # 각 P에 대해 num_samples 개의 샘플 생성
    for _ in range(num_samples):
        # h 생성 및 정렬
        h = np.abs(np.random.normal(0, 1, K))
        h_ordered = np.sort(h)

        #### MSE, PW, ESR calculation ####
        # OAC policy
        MSE_OAC, PW_OAC, i_star = oac.OAC(K, P, sigma, h_ordered)
        ESR_OAC = 100 - PW_OAC / (K * P) * 100  # energy efficiency

        # CH inversion policy
        MSE_OAC_CH_inversion, PW_OAC_CH_inversion = oac.OAC_CH_inversion(K, P, sigma, h_ordered)
        ESR_OAC_CH_inversion = 100 - PW_OAC_CH_inversion / (K * P) * 100

        # Energy greedy policy (OAC_Energy_greedy)
        MSE_OAC_Energy_greedy, PW_OAC_Energy_greedy = oac.OAC_Energy_greedy(K, P, sigma, h_ordered)
        ESR_OAC_Energy_greedy = 100 - PW_OAC_Energy_greedy / (K * P) * 100

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

    #### append to list ####
    # OAC policy
    MSE_OAC_list.append(MSE_OAC_avg)
    PW_OAC_list.append(PW_OAC_avg)
    ESR_OAC_list.append(ESR_OAC_avg)

    # CH inversion policy
    MSE_OAC_CH_inversion_list.append(MSE_OAC_CH_inversion_avg)
    PW_OAC_CH_inversion_list.append(PW_OAC_CH_inversion_avg)
    ESR_OAC_CH_inversion_list.append(ESR_OAC_CH_inversion_avg)

    # Energy greedy policy
    MSE_OAC_Energy_greedy_list.append(MSE_OAC_Energy_greedy_avg)
    PW_OAC_Energy_greedy_list.append(PW_OAC_Energy_greedy_avg)
    ESR_OAC_Energy_greedy_list.append(ESR_OAC_Energy_greedy_avg)

# 첫 번째 그래프 출력 (MSE_OAC vs P(dB))
plt.figure(figsize=(10, 5))
plt.plot(P_dB, MSE_OAC_list, marker='o', linestyle='-', color='r', label='OAC')
plt.plot(P_dB, MSE_OAC_CH_inversion_list, marker='v', linestyle='-', color='g', label='CH Inversion')
plt.plot(P_dB, MSE_OAC_Energy_greedy_list, marker='s', linestyle='-', color='b', label='Energy Greedy')
plt.title('Average MSE_OAC vs P (in dB)')
plt.xlabel('P (dB)')
plt.ylabel('Average MSE_OAC')
plt.ylim(0,K/2)
plt.legend()
plt.grid(True)
plt.show()

# 두 번째 그래프 출력 (ESR_OAC vs P(dB))
plt.figure(figsize=(10, 5))
plt.plot(P_dB, ESR_OAC_list, marker='o', linestyle='-', color='r', label='OAC')
plt.plot(P_dB, ESR_OAC_CH_inversion_list, marker='v', linestyle='-', color='g', label='CH Inversion')
plt.plot(P_dB, ESR_OAC_Energy_greedy_list, marker='s', linestyle='-', color='b', label='Energy Greedy')
plt.title('Average ESR_OAC vs P (in dB)')
plt.xlabel('P (dB)')
plt.ylabel('Average ESR_OAC [%]')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()
