from sklearn.datasets import load_breast_cancer, load_boston
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)
# torch :  1.10.2 사용DEVICE :  cuda

#1. 데이터
# datasets = load_breast_cancer()
datasets = load_boston()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape) # (354, 13) torch.Size([354, 1])
print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    # nn.Sigmoid(),
).to(DEVICE)

#3. 컴파일, 훈련
# criterion = nn.BCELoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
        
    loss.backward()     # 역전파
    optimizer.step()
    return loss.item()
    
EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {:.8f}'.format(epoch, loss))

#4. 평가, 예측
print("================ 평가, 예측 ================")    
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()

loss = evaluate(model, criterion, x_test, y_test)  
print('loss : ', loss)  

# y_predict = (model(x_test) >= 0.5).float()
# print(y_predict[:10])
y_predict = model(x_test)

# score = (y_predict == y_test).float().mean()
# print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
# score = accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())
# print('accuracy : {:.4f}'.format(score))
score = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print('R2 score : {:.4f}'.format(score))

# ================ 평가, 예측 ================
# loss :  7.757288932800293
# R2 score : 0.9061