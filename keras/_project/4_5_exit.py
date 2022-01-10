import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.models import Sequential
data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")
# print(type(data1))
# print(type(data2))

dataset = pd.concat([data1,data2],ignore_index=True)
del dataset['Unnamed: 0']
# print(dataset)
# dataset.to_csv('data_reset.csv', index=True, encoding='utf-8-sig')

print(dataset.info())
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


# print(np.unique(y))    # [0 1] 

# print(x.shape)   # (36, 55)
# print(y.shape)   # (36,)

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.8, shuffle=True, random_state=49)

# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=55))
# model.add(Dense(48))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(18))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()


# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

# model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) 


# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ',loss)

# # resulte = model.predict(x_test[:31])
# #print(y_test[:31])
# #print(resulte)


print("바보")