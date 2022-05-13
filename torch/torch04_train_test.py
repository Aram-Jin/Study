from pickletools import optimize
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, ' 사용DEVICE :', DEVICE)

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.size())

#2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 4),
    nn.Linear(4, 1),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# print(optimizer)
def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()#!
    
    hyporthesis = model(x)
    loss = criterion(hyporthesis, y)
    
    loss.backward()  #!
    optimizer.step() #!
    
    return loss.item()

EPOCHS = 100
for epoch in range(1,EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch : ",epoch, "loss : ",loss)   
    
#4. 평가, 예측
# loss = model.evaluate(x, y)    
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        predict = model(x_test)
        loss2 = criterion(predict, y_test)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss: ', loss2)

result = model(x_test.to(DEVICE))
print('result: ', result.cpu().detach().numpy())

print('---------------------------------------')
predict_value = torch.Tensor(([[11]])).to(DEVICE)
result = model(predict_value).to(DEVICE)

print(result.cpu().detach().numpy())

# tensor를 numpy로 변환
