# Logistic Regression -> Binary (0 or 1 (Facebook Feed for user preference))
# 0 또는 1 중에 어느 값에 가까운지
# H(x) = P(x = 1; w) x가 1일 확률
#      = 1 - P(x = 0; w)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
torch.manual_seed(1)

x_data = [[1, 2],[2, 3],[3, 1],[4, 3],[5, 3],[6, 2]] # input 6 X 2
y_data = [[0],[0],[0],[1],[1],[1]] # output 6 X 1 ( (6 X 2 ) X (2 X 1) )

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros([2,1], requires_grad=True) # Weight 2 X 1
b = torch.zeros(1,requires_grad=True) # bias 1
optimizer = optim.SGD([W,b], lr = 1) # SGD를 가지고 W, b를 학습하는데 learing rate는 1이야
np_epochs = 1000
for epoch in range(np_epochs + 1):

    # hypothesis = 1 / (1 + torch.exp(-(torch.matmul(x_train, W) + b)))
    hypothesis = torch.sigmoid(torch.matmul(x_train, W) + b) # 파이토치에서 제공하는 sigmoid 함수 사용
    # losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log((1-hypothesis)))
    # cost = losses.mean()
    cost = F.binary_cross_entropy(hypothesis, y_train) # 위 두줄을 파이토치에서 제공하는 함수 사용

    optimizer.zero_grad() # 혹시나 기존에 gradient를 구한게 있으면 0으로 초기화, 안하면 기존에 값에 계속 더함
    cost.backward() # 이제까지 연산에 사용했던 W, b는 gradient가 구해져 있음
    optimizer.step() # 이걸가지고 cost를 줄이는 방향으로 W, b를 Update

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, np_epochs, cost.item()))


# 실전 - Higher Implementation with Class
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = BinaryClassifier()


optimizer = optim.SGD(model.parameters(),lr = 1)
np_epochs = 100
for epoch in range(np_epochs + 1):

    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # hypothesis가 0.5보다 크면 1
        correct_prediction =prediction.float() == y_train
        # prediciton은 ByteTensor인데 이걸 FloatTensor로 바꾸고 y_train과 비교해서 같으면 1
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost {:6f} Accuracy {:2.2f}%'.format(
            epoch,np_epochs,cost.item(),accuracy * 100,
        ))