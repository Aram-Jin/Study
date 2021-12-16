import numpy as np, time
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)    # (178, 13) (178,)
# print(np.unique(y,return_counts='True'))   # [0 1 2]  (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

y = to_categorical(y)
# print(y.shape)   # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0],x_train.shape[1],1)
x_test = scaler.fit_transform(x_test).reshape(x_test.shape[0],x_test.shape[1],1)
# print(x_train.shape, x_test.shape)   # (142, 13, 1) (36, 13, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(13,1)))
model.add(Flatten())
model.add(Dropout(0.2))               
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(3, activation='softmax'))
# model.summary()   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=False)

start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.2)  # , callbacks=[es]
end = time.time() - start

print("걸린시간: ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss[0])
print('accuracy :',loss[1])

y_predict = model.predict(x_test)

'''
걸린시간:  9.229 초
loss : 0.013855204917490482
accuracy : 0.9444444179534912

< EarlyStopping 없이 모든 epochs 돌렸을때 연산속도 비교(LSTM vs Conv1D) >

걸린시간:  15.265 초
loss : 0.025034435093402863
accuracy : 0.9444444179534912
==========================================
걸린시간:  9.266 초
loss : 0.013928375206887722
accuracy : 0.9722222089767456

'''