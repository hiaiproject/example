import numpy as np
import torch

# 입력 : x_train
# 출력 : y_train
# 입력과 출력을 별개의 텐서로 구분

x_train = torch.FloatTensor([[1], [2], [3]])  # 입력
y_train = torch.FloatTensor([[2], [4], [6]])  # 출력

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# y = Wx + b
# Weight와 bias를 0으로 초기화
# requires_grad=True로 학습할 것이라고 명시

# Gradient Descent
optimizer = torch.optim.SGD([W, b], lr=0.01) #[W,b]는 학습할 텐서들, lr은 Learning Rate

nb_epochs = 1000

for epoch in range(1, nb_epochs + 1):
    # Hypothesis
    hypothesis = W * x_train + b # Linear Regression

    # Cost function
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 항상 붙어다니는 3줄
    optimizer.zero_grad()  # gradient 초기화
    cost.backward()  # gradient 계산
    optimizer.step()  # 개선
# 반복하며 Hypothesis 예측 -> Cost 계산 -> Optimizer로 학습



