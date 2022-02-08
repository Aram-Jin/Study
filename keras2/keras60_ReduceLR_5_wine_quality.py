import time
import numpy as np
import pandas as pd
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
path = "../_data/"    

data = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';', dtype=float)
# print(data.shape)  # (4898, 12)
print(data.head())
print(data.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
print(data.info())
# print(data.columns)
# print(type(data))   # <class 'pandas.core.frame.DataFrame'>

data = data.values   # numpy형태로 변경하는 방법
print(type(data))   # <class 'numpy.ndarray'>
print(data.shape)  # (4898, 12)

# numpy형태에서 x,y나누기 
x = data[:, :11]
y = data[:, 11]

# x = data.drop(['quality'], axis=1)  
# # print(x.shape)  # (4898, 11)
# y = data['quality']
# # print(y.shape)  # (4898,)

print(np.unique(y, return_counts=True))  # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# le = LabelEncoder()
# le.fit(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print("걸린시간: ", end - start)



'''
results:  0.7102
accuracy_score :  0.7102
걸린시간:  38.32083296775818

results:  0.6745
accuracy_score :  0.6745
f1_score :  0.4523239916472014
걸린시간:  24.807340145111084
'''
