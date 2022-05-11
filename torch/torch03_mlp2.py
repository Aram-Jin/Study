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
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]])

y = np.array([11,12,13,14,15,16,17,18,19,20])

x = np.transpose(x) # (3, 10) -> (10, 3)

x = torch.FloatTensor(x).to(DEVICE)    
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   # (10,) -> (10,1)    => 데이터를 텐서형태로 만들고 reshape해주기(행렬형태로)

print(x,y)      
print(x.shape, y.shape)  # torch.Size([10, 3]) torch.Size([10, 1])

#2. 모델구성
# model = nn.Linear(1, 1).to(DEVICE)   # 앞에 1은 input, 뒤에 1이 output
model = nn.Sequential(
    nn.Linear(3, 5),       # 인풋은 행무시!
    nn.Linear(5, 3), 
    nn.Linear(3, 4), 
    nn.Linear(4, 2), 
    nn.Linear(2, 1), 
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

print(optimizer)

def train(model, criterion, optimizer, x, y):
    # model.train()    # 훈련모드
    optimizer.zero_grad()  # 가중치 초기화
    
    hypothesis = model(x)  
    
    loss =  criterion(hypothesis, y)    
    # loss = nn.MSELoss()(hypothesis, y)  # 에러
    # loss = nn.MSELoss()(hypothesis, y)
    # loss = F.mse_loss(hypothesis, y)
    
    loss.backward()    # 기울기값 계산까지다
    optimizer.step()   # 가중치 수정
    return loss.item()  # 그냥반환하면 tensor형태이므로 .item()으로 사용할 수 있는 형태로 반환

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

result = model(torch.Tensor([[10, 1.3, 1]]).to(DEVICE))
print('[10, 1.3, 1]의 예측값 : ', result.item())

# 최종 loss :  2.8774776339446362e-08
# [10, 1.3, 1]의 예측값 :  20.000083923339844