import pandas as pd
import numpy as np
import requests
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Input
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

print(x.shape)   # (138, 55)
print(y.shape)   # (138,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(48, activation='tanh', input_dim=55))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=5, verbose=1, validation_split=0.2, callbacks=[es]) 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ',loss[1])

resulte = model.predict(pre_data)

print(resulte[0])
print(resulte[1])
print(resulte[2])

# cm =confusion_matrix(y_train)
# print(cm)


# {'safe': 0, 'danger': 1}

for i in resulte:

    if( i <=0.5):
        safe = resulte[0][0]*100 
        print(f"이 종목의 위험률은 {round(safe,2)} % 입니다. 투자해도 좋습니다.")
    elif( i >=0.5):
        danger = 100 - resulte[0][0]*100
        print(f"이 종목은 {round(danger,2)} % 확률로 관리종목으로 지정될 가능성이 있는 위험한 종목입니다.")
    else:
        print("ERROR")
        

'''
loss :  0.840522050857544
accuracy :  0.6363636255264282
[0.85560787]
[0.00167173]
이 종목은 85.56 % 확률로 관리종목으로 지정될 가능성이 있는 위험한 종목입니다.
이 종목은 14.44 % 확률로 투자해도 안전한 종목입니다.
'''

# 아센디오'012170' 
# 신풍제약'019170','001440'
# pre_code = ['012170','019170','001440']
