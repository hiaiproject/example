import numpy as np
import torch

t = np.array([[[0, 1, 2],
               [3, 4, 5]],  # 3 X 2 -> 밑변

              [[6, 7, 8],
               [9, 10, 11]]])  # (3 X 2) X 2 -> 높이가 2

ft = torch.FloatTensor(t)
print(ft.shape)
# 첫번째 차원이 [[0,1,2],[3,4,5]] / [[6,7,8][9,10,11]] -> 2게
# 두번째 차원이 [0,1,2] / [3,4,5] -> 2개
# 세번째 차원이 0 / 1 / 2  -> 3개
# 따라서 shape는 [2, 2, 3]


# View : 텐서를 원하는 Shape로 자유자재로 바꿀수 있다
print(ft.view([-1, 3]))
# 첫번째 차원은 모르겠고 두번째 차원이 3개의 레벨을 갖도록 ft를 재설정
# |ft| = (2, 2 ,3) -> (2x2, 3) -> (4, 3) 4 X 3 행렬으로 재설정 됨
print(ft.view([-1, 3]).shape)
# shape가 (4, 3)으로 재설정 되는데 이 숫자의 곱들이 기존 행렬의 숫자의 곱과 같기만 하면 됨

print(ft.view([-1, 1, 3]))  # 4 X 1 X 3 array가 된다
print(ft.view([-1, 1, 3]).shape)

# Squeeze : dimension을 없애줌 (쥐어 짜기)
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)  # 크기가 1인 dimension을 없애준다
# print(ft.squeeze(n)) 이면 n번째 차원의 크기가 1이면 없애준다

# UnSqueeze : 내가 원하는 dimension에 1을 넣어줌
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.unsqueeze(-1))  # 마지막 dimension (3,) -> (3, 1)

# Type Casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float())  # [1., 2., 3., 4.]

bt = torch.ByteTensor([True, False, True, False])
print(bt)
print(bt.long())  # [1,0,0,1]
print(bt.float())  # [1., 0., 0., 1.]

x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])

# Concatenation
print(torch.cat([x,y], dim = 0))
# x, y를 dimension 0에 대해 concatenation / 2 X 2 -> 4 X 2
print(torch.cat([x,y], dim = 1))
# x, y를 dimension 1에 대해 concatenation / 2 X 2 -> 2 X 4

# Stacking : Concatination을 편리하게, 텐서들을 쌓는다는 개념으로 이해
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim = 1))

# Ones and Zeros : 1로만 가득 채우거나 0으로만 가득 채우거나
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))

# In-place Operation : _가 붙은 곱하기

x = torch.FloatTensor([[1,2], [3,4]])
print(x.mul(2.)) # 새로운 메모리에 결과값을 넣는다
print(x)
print(x.mul_(2.)) # 메모리를 새로 할당하지 않고 기존의 텐서값에 넣는다
print(x)

