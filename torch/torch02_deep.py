from unittest import result
import numpy as np
import torch
import torch.nn as nn   #nn : newral network
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용DEVICE :', DEVICE)
# torch :  1.10.2  사용DEVICE : cuda

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)    # (3,) -> (3, 1) 
y = torch.FloatTensor(y).to(DEVICE)    # (3,) -> (3, 1)   => 데이터를 텐서형태로 만들고 reshape해주기(행렬형태로)

print(x,y)       # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape, y.shape)   # torch.Size([3]) torch.Size([3])

#2. 모델구성
# model = nn.Linear(1, 1).to(DEVICE)   # 앞에 1은 input, 뒤에 1이 output
model = nn.Sequential(
    nn.Linear(1, 5), 
    nn.Linear(5, 3), 
    nn.Linear(3, 4), 
    nn.Linear(4, 2), 
    nn.Linear(2, 1), 
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(optimizer)


def train(model, criterion, optimizer, x, y):
    # model.train()    # 훈련모드
    optimizer.zero_grad()  # 가중치 초기화
    
    hypothesis = model(x)  
    
    # loss =  criterion(hypothesis, y)    # 요기까지 순전파
    # loss = nn.MSELoss()(hypothesis, y)  # 에러
    # loss = nn.MSELoss()(hypothesis, y)
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward()    # 기울기값 계산까지다
    optimizer.step()   # 가중치 수정
    return loss.item()

epochs = 10000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))

print("=================================================")
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()   # 평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss = criterion(predict, y)
    return loss.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

# result = model.predict([4])

result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 : ', result.item())


# 최종 loss :  9.473903595218908e-14
# 4의 예측값 :  4.000000476837158


