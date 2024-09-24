import numpy as np
import matplotlib.pyplot as plt
import OAC as oac

# 초기 설정
K = 5
sigma = 1  # sigma^2
num_samples = 1000  # 각 P마다 100개의 샘플 생성

# dB 단위에서의 P 설정 (0dB, 5dB, 10dB, 15dB, 20dB)
P_dB = np.arange(0, 25, 2.5)
P_values = 10 ** (P_dB / 10)  # dB 값을 선형 값으로 변환

# 결과 저장용 리스트
MSE_OAC_list = []
PW_OAC_list = []
ESR_OAC_list = [] # energy savings rate

# P 값을 변화시키며 MSE_OAC와 PW_OAC 계산
for P in P_values:
    MSE_OAC_samples = []
    PW_OAC_samples = []
    ESR_OAC_samples = []

    # 각 P에 대해 100개의 샘플 생성
    for _ in range(num_samples):
        # h 생성 및 정렬
        h = np.abs(np.random.normal(0, 1, K))
        h_ordered = np.sort(h)

        # OAC 알고리즘 적용
        MSE_OAC, PW_OAC, i_star = oac.OAC(K, P, sigma, h_ordered)
        ESR = 100 - PW_OAC / (K * P) * 100 # energy efficiency

        # 샘플의 MSE_OAC[i_star]와 PW_OAC 저장
        MSE_OAC_samples.append(MSE_OAC[i_star])
        PW_OAC_samples.append(PW_OAC)
        ESR_OAC_samples.append(ESR)

    # 100개의 샘플 평균 계산
    MSE_OAC_avg = np.mean(MSE_OAC_samples)
    PW_OAC_avg = np.mean(PW_OAC_samples)
    ESR_OAC_avg = np.mean(ESR_OAC_samples)

    # 평균값을 리스트에 저장
    MSE_OAC_list.append(MSE_OAC_avg)
    PW_OAC_list.append(PW_OAC_avg)
    ESR_OAC_list.append(ESR_OAC_avg)

# 첫 번째 그래프 출력 (MSE_OAC vs P(dB))
plt.figure(figsize=(10, 5))
plt.plot(P_dB, MSE_OAC_list, marker='o', linestyle='-', color='r')
plt.title('Average MSE_OAC vs P (in dB)')
plt.xlabel('P (dB)')
plt.ylabel('Average MSE_OAC')
plt.grid(True)
plt.show()

# 두 번째 그래프 출력 (PW_OAC vs P(dB)
# plt.figure(figsize=(10, 5))
# plt.plot(P_dB, PW_OAC_list, marker='o', linestyle='-', color='g')
# plt.title('Average PW_OAC vs P (in dB)')
# plt.xlabel('P (dB)')
# plt.ylabel('Average PW_OAC')
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(P_dB, ESR_OAC_list, marker='o', linestyle='-', color='b')
plt.title('Average ESR_OAC vs P (in dB)')
plt.xlabel('P (dB)')
plt.ylabel('Average ESR_OAC [%]')
plt.ylim(0, 100)
plt.grid(True)
plt.show()
