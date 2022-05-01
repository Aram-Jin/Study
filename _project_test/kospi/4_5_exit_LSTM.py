import pandas as pd
import numpy as np
import requests
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout
data1 = pd.read_csv("./관리종목18data.csv")
data2 = pd.read_csv("./안전종목120data.csv")
data3 = pd.read_csv("./평가종목3data.csv")

# print(type(data1))
# print(type(data2))

dataset = pd.concat([data1,data2],ignore_index=True).astype('float')
pre_data = data3.astype('float')
# print(type(pre_data))
del dataset['Unnamed: 0']
del pre_data['Unnamed: 0']

# print(type(dataset))
# print(type(pre_data))
# print(dataset.info())
# print(pre_data.info())

# print(dataset)

# dataset.to_csv('dataset_final.csv', index=True, encoding='utf-8-sig')

# print(dataset.info())
# print(dataset.feature_names)
# print(dataset.DESCR)
# print(np.min(dataset), np.max(dataset))
# print(dataset.head)
# for col in dataset.columns:
#     print(col)
# print(dataset.index)
# np.array(dataset)

x = dataset.drop(['Target'], axis=1)
y = dataset['Target']

# x = np.log1p(x)

# print(np.unique(y))    # [0 1] 
x = np.array(x)
print(x.shape)   # (138, 55)
print(y.shape)   # (138,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

# print(x_train.shape)  # (110, 55, 1)
# print(x_test.shape)  # (28, 55, 1)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)

x = x.reshape(138, 11,5)
x_train = scaler.fit_transform(x_train).reshape(len(x_train),11,5)
x_test = scaler.transform(x_test).reshape(len(x_test),11,5)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(11,5))) 
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(48, activation='relu'))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(24))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100000, batch_size=10, verbose=1, validation_split=0.2, callbacks=[es]) 


#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ',loss[1])

#5. 예측 ->  예측하고자 하는 종목 수량만큼 result를 출력해줄 수 있음
# 아센디오(012170) -> 2021.03 최근 관리종목에서 해지된 종목
# 신풍제약(019170), 대한전선(001440) -> 시총상위 150위 내 종목 

pre_data = np.array(pre_data)
pre_data = pre_data.reshape(len(pre_data),11,5)
result = model.predict(pre_data)
result = result*100
print(result[0])
print(result[1])
print(result[2])

# {'safe': 0, 'danger': 1}
for i in result:
    if( i <=50):
        safe = i
        print(f"이 종목은 투자해도 좋습니다. 위험률 {np.round(safe,2)} %  ")
    elif( i >50):
        danger = i
        print(f"이 종목은 위험률 {np.round(danger,2)} % 으로 관리종목으로 지정될 가능성이 있는 위험한 종목입니다.")
    else:
        print("ERROR")
        

'''
# 아센디오(012170) 
# 신풍제약(019170),대한전선(001440)
# pre_code = ['012170','019170','001440']

loss :  0.3151543438434601
accuracy :  0.9285714030265808
[69.83464]
[4.8694215]
[3.148824]
이 종목은 위험률 [69.83] % 으로 관리종목으로 지정될 가능성이 있는 위험한 종목입니다.
이 종목은 투자해도 좋습니다. 위험률 [4.87] %
이 종목은 투자해도 좋습니다. 위험률 [3.15] %
'''

'''
Epoch 00112: early stopping
1/1 [==============================] - 0s 16ms/step - loss: 0.3003 - accuracy: 0.9286
loss :  0.30030760169029236
accuracy :  0.9285714030265808
[80.316185]
[1.7590599]
[0.8316596]
이 종목은 관리종목으로 지정될 가능성이 있는 위험한 종목입니다. 위험도 [80.32] %
이 종목은 투자해도 좋습니다. 안전도 [98.24] %
이 종목은 투자해도 좋습니다. 안전도 [99.17] %
'''





# stateful='True', 