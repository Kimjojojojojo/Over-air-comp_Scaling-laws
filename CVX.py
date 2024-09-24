import cvxpy as cp
import numpy as np

# 주어진 변수
K = 5  # 예를 들어 K = 5
h = np.random.rand(K)  # 임의의 h_k 값 (양의 실수)
sigma2 = 1  # 예를 들어 잡음 sigma^2
P = 10  # 제약 조건 상수 P

# 최적화 변수 정의 (a와 b는 모두 양의 실수)
a = cp.Variable(nonneg=True)
b = cp.Variable(K, nonneg=True)

# MSE 표현식 정의
MSE = cp.sum_squares(a * h * b - 1) + sigma2 * cp.square(a)

# 제약 조건 정의
constraints = [cp.square(b) <= P]

# 목적 함수: MSE를 최소화
objective = cp.Minimize(MSE)

# 최적화 문제 정의 및 풀이
prob = cp.Problem(objective, constraints)
prob.solve()

# 결과 출력
print("Optimal value of a:", a.value)
print("Optimal values of b_k:", b.value)
print("Optimal MSE:", prob.value)
