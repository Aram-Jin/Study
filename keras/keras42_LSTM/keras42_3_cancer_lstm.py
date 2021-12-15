import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
#print(datasets.DESCR)     # DESCR: 데이터셋에 대한 간략한 설명
#print(datasets.feature_names)

x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    # (569, 30) (569,)
#print(np.unique(y))   # [0 1]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)
#print(x_train.shape, x_test.shape)    # (455, 30) (114, 30)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(455,30,1)
x_test = scaler.fit_transform(x_test).reshape(114,30,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(40, return_sequences=False, input_shape=(30,1)))
model.add(Dropout(0.2))  
model.add(Dense(30))
model.add(Dropout(0.2))                  
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
#model.summary()   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss[0])
print('accuracy :',loss[1])

y_predict = model.predict(x_test)

'''
loss : 0.033136893063783646
accuracy : 0.9561403393745422
'''