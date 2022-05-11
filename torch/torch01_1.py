from unittest import result
import numpy as np
import torch
import torch.nn as nn   #nn : newral network
import torch.optim as optim
import torch.nn.functional as F

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)    # (3,) -> (3, 1) 
y = torch.FloatTensor(y).unsqueeze(1)    # (3,) -> (3, 1)   => 데이터를 텐서형태로 만들고 reshape해주기(행렬형태로)

print(x,y)       # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape, y.shape)   # torch.Size([3]) torch.Size([3])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))  # 인풋은 행무시!
model = nn.Linear(1, 1)  # 앞에 1은 input, 뒤에 1이 output   


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(optimizer)

# model.fit(x, y, epochs=100, batch_size=1)
def train(model, criterion, optimizer, x, y):    # zero_grad() backward() step() 암기!!!
    # model.train()    # 훈련모드
    optimizer.zero_grad()  # 가중치 초기화
    
    hypothesis = model(x)  
    
    loss =  criterion(hypothesis, y)    # 요기까지 순전파
    
    loss.backward()    # 기울기값 계산까지다
    optimizer.step()   # 가중치 수정
    return loss.item()

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))

print("=================================================")
    
#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):     
    model.eval()   # 평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss = criterion(predict, y)
    return loss.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

# result = model.predict([4])

result = model(torch.Tensor([[4]]))
print('4의 예측값 : ', result.item())


# 최종 loss :  4.735739821626339e-06
# 4의 예측값 :  3.995602607727051


