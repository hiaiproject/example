import numpy as ns
import torch


t = ns.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('Rank of t: ', t.ndim)  # dimension개의 차원
print('Shape of t: ', t.shape)  # 각각의 차원에 존재하는 element 수

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])  # Element
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])  # Slicing t[2부터 4까지], t[4부터 -1(마지막)전까지]
print('t[:2] t[3:]  = ', t[:2], t[3:])  # Slicing t[0부터 까지], t[3부터 마지막까지]


t = ns.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9],[10.,11.,12.]]) # 2D array with NumPy
print(t)
print('Rank of t: ', t.ndim)  # dimension개의 차원
print('Shape of t: ', t.shape)  
# 각각의 차원에 shape개의 element 존재
# (4 ,3) 이면 첫번째 차원에 4개의 element 존재, 두번째 차원에 3개의 element 존재

t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t.dim()) # 몇개의 차원이 있니
print(t.shape) # 각 차원의 모양은 어떠니
print(t.size()) # 각 차원의 크기는 어떠니(= shape()와 같은 값)
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])
print('t[:2] t[3:]  = ', t[:2], t[3:])

t = torch.FloatTensor([[1.,2.,3.], [4.,5.,6.], [7.,8.,9],[10.,11.,12.]]) # 4 x 3 의 Matrix
print(t)


print(t.dim()) #rank
print(t.size()) #shape
print(t[:, 1]) #첫번째 차원에서는 다 가져오되 두번째 차원에서는 index가 1에 해당하는 값만 가져와라
print(t[:, 1].size()) #t[:, -1]은 size()개의 element를 가진 vector (4개의 element를 가진 벡터야)
print(t[:, :-1]) #첫번째 차원에서는 다 가져오되 두번째 차원에서는 처음부터 마지막 전까지 해당하는 값만 가져와라
print(t[:,:-1].size()) #그의 크기(=모양)


# BroadCasting : 자동적으로 size를 맞춰서 진행

# 덧셈이나 뺄셈을 할땐 두 텐서의 크기가 같아야 한다
# Same Shape
m1 = torch.FloatTensor([[3, 3]])  # 1 X 2 Matrix
m2 = torch.FloatTensor([[2, 2]])  # |m1| = (1,2) = |m2|
print(m1 + m2)

# Vector + Scalar
m1 = torch.FloatTensor([[1, 2]])  # 1 X 2 Matrix
m2 = torch.FloatTensor([3])  # 3 (Scalar) -> [[3,3]] (1 X 2 Vector) 같은 사이즈로 만들어 주어서 쉽게 연산
print(m1 + m2)

# 2 X 1 Vector + 1 X 2 Vector
m1 = torch.FloatTensor([[1, 2]])  # 2 X 2 로 [1,2],[1,2]
m2 = torch.FloatTensor([[3], [4]])  # 2 X 2 로 [3,3],[4,4]
print(m1 + m2)

# Multiplication vs Matrix Multiplication
# -> BroadCasting으로 인해 잘못된 값을 나올수 있으므로 주의!!

# 딥러닝은 행렬곱 연산을 굉장히 많이 사용하는 알고리즘

print()
print("=============")
print("Mul vs MatMul")
print("=============")
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix M1', m1.shape)  # 2 X 2
print('Shape of Matrix M2', m2.shape)  # 2 X 1
print(m1.matmul(m2))  # 2 X 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix M1', m1.shape)  # 2 X 2
print('Shape of Matrix M2', m2.shape)  # 2 X 1 -> 2 X 2 ([1,1],[2,2])로 브로드캐스팅
print(m1 * m2)  # 2 X 2
print(m1.mul(m2))  # 2 X 2

t = torch.FloatTensor([1, 2])
print(t.mean())  # mean()은 평균을 계산

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean())
print(t.mean(dim=0))
# dim 0 을 없애겠어 (행렬에서 세로방향으로 접는다) 2 X 2 -> 1 X 2
# [1 2],[3 4]에서 [1 + 3 / 2 , 2 + 4 / 2]
print(t.mean(dim=1))
# dim 1 을 없애겠어 (행렬에서 가로방향으로 접는다) 2 X 2 -> 2 X 1

# Sum()에 대해서도 Mean()과 동일
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print('\n')

#Max()는 가장 큰 값을 찾아주고 Argmax()는 그 index를 리턴
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.max(dim=0)) #max와 그 index를 함께 출력
print(t.max(dim=0)[0])
print(t.max(dim=0)[1])
print('\n')
print(t.max(dim=1)) #max와 그 index를 함께 출력
print(t.max(dim=1)[0])
print(t.max(dim=1)[1])