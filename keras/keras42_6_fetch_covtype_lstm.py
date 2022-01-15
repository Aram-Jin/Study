import numpy as np, time
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
model.add(LSTM(32, return_sequences=False, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))               
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(y.shape[1], activation='softmax'))
# model.summary()   


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=False)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=500, validation_split=0.2, callbacks=[es]) 
end = time.time() - start

print("걸린시간: ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss[0])
print('accuracy :',loss[1])

y_predict = model.predict(x_test)


'''
loss : 0.04008910432457924
accuracy : 0.8097209334373474
'''