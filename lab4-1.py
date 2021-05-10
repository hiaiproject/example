import torch
import math

# 학습할 date set이 1개
x_train = torch.FloatTensor([[1], [2], [3]])  # 입력
y_train = torch.FloatTensor([[1], [2], [3]])  # 출력
# H(x) = wx + b

# 학습할 date set이 여러개
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 80],
                            [96, 98, 100],
                            [73, 66, 70]])  # 입력
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])  # 출력

# H(x) = w1x1 + w2x2 + w3x3 + b -> matrix를 사용

W = torch.zeros([3, 1], requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=1e-5) # [W]는 학습할 텐서들, lr은 Learning Rate


nb_epochs = 20

for epoch in range(nb_epochs + 1):
    # Hypothesis
    hypothesis = x_train.matmul(W) + b

    # Cost Gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2) # MSE

    print('Epoch {:4d}/{} Hypothesis: {}, Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(),cost.item()
    ))


    # 항상 붙어다니는 3줄, cost로 H(x)개선
    optimizer.zero_grad()  # gradient 초기화
    cost.backward()  # gradient 계산
    optimizer.step()  # gradient descent
# 반복하며 Hypothesis 예측 -> Cost 계산 -> Optimizer로 학습

# 위 예제에서 W가 1일때 cost = 0
# 1에서 멀어질수록 cost가 높아짐

