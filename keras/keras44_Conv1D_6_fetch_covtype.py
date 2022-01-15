import numpy as np, time
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = fetch_covtype()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  # (581012, 54) (581012,)
# print(np.unique(y, return_counts='True'))  # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301, 35754, 2747, 9493, 17367, 20510], dtype=int64))

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y.reshape(-1,1)) 
# print(y.shape)   # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0],x_train.shape[1],1)
x_test = scaler.fit_transform(x_test).reshape(x_test.shape[0],x_test.shape[1],1)
print(x_train.shape, x_test.shape)   # (464809, 54, 1) (116203, 54, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(x_train.shape[1],1)))
model.add(Flatten())
model.add(Dropout(0.2))               
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(y.shape[1], activation='softmax'))
# model.summary()   


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
# es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=False)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=500, validation_split=0.2) #, callbacks=[es]
end = time.time() - start

print("걸린시간: ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss[0])
print('accuracy :',loss[1])

y_predict = model.predict(x_test)



'''
걸린시간:  598.723 초
loss : 0.05454786866903305
accuracy : 0.7270724773406982


< EarlyStopping 없이 모든 epochs 돌렸을때 연산속도 비교(LSTM vs Conv1D) >

걸린시간:  2317.549 초
loss : 0.03704901039600372
accuracy : 0.8266310095787048
=======================================================================
걸린시간:  590.165 초
loss : 0.054358676075935364
accuracy : 0.727941632270813

'''