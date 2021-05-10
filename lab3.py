import numpy as np
import torch

# 입력 : x_train
# 출력 : y_train
# 입력과 출력을 별개의 텐서로 구분

x_train = torch.FloatTensor([[1], [2], [3]])  # 입력
y_train = torch.FloatTensor([[1], [2], [3]])  # 출력

W = torch.zeros(1, requires_grad=True)
# y = Wx + b
# Weight를 0으로 초기화
# requires_grad=True로 학습할 것이라고 명시

# Gradient Descent
optimizer = torch.optim.SGD([W], lr=0.15) #[W]는 학습할 텐서들, lr은 Learning Rate
nb_epochs = 10

for epoch in range(nb_epochs + 1):
    # Hypothesis
    hypothesis = W * x_train # Linear Regression

    # Cost Gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2) # MSE

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(),cost.item()
    ))


    # 항상 붙어다니는 3줄, cost로 H(x)개선
    optimizer.zero_grad()  # gradient 초기화
    cost.backward()  # gradient 계산
    optimizer.step()  # gradient descent


# 반복하며 Hypothesis 예측 -> Cost 계산 -> Optimizer로 학습

# 위 예제에서 W가 1일때 cost = 0
# 1에서 멀어질수록 cost가 높아짐


