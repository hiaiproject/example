# Softmax Classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# P(주먹|가위) : 이전에 가위를 내고 주먹을 낼 확률
# P(가위|가위) : 이전에 가위를 내고 가위를 낼 확률
# P(보|가위) : 이전에 가위를 내고 보를 낼 확률
z = torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z, dim=0) # max값을 뽑는데 부드럽게(합이 1이 되게) 뽑음
print(hypothesis)
print(hypothesis.sum())

z  = torch.rand(3,5,requires_grad= True) # |Z| = 3 X 5
hypothesis = F.softmax(z, dim=1) # 2번째 dimension에 대해 softmax를 수행해
print(hypothesis) # Prediction(y-hat)

y = torch.randint(5, (3,)).long()
print(y) # 대충 설정한 y (실제 정답의 index 값)

y_one_hat = torch.zeros_like(hypothesis) # hypothesis와 같은 크기의 벡터를 만들어 줌
y_one_hat.scatter_(1, y.unsqueeze(1), 1) # inplace(_ 메모리를 새로할당 x)
# dim = 1에 대해서 y.unsqueeze(1) (0, 2, 1)을 가지고 1을 뿌려줘
#[1 0 0 0 0]
#[0 0 1 0 0]
#[0 1 0 0 0]

cost = (y_one_hat * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)