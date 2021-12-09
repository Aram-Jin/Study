from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   
import numpy as np, datetime, pandas as pd

"""
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape)   # (442, 10) (442,)   ---> (442, 2, 5, 1)
#print(y.shape)   # (442,)

x = x.reshape(x.shape[0], 2, 5, 1)  
#print(x.shape)   # (442, 2, 5, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), padding='same', strides=1, input_shape=(2, 5, 1)))  
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(1))
#model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'                
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   
model_path = "".join([filepath, 'k35_diabetes_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, callbacks=[es, mcp])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#loss : 3413.446044921875
#r2스코어 :  0.4740486201574936
"""

########################################## column drop ##########################################

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape)   # (442, 10) (442,)   
#print(y.shape)   # (442,)

xx = pd.DataFrame(x, columns=datasets.feature_names)
#print(type(xx))   # <class 'pandas.core.frame.DataFrame'>
#print(xx)
#print(xx.corr())  

xx['diabetes'] = y 
#print(xx)
#print(xx.corr())   

#import matplotlib.pyplot as plt
#import seaborn as sns  
#plt.figure(figsize=(10,10))
#sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True) 
#plt.show()

x = xx.drop(['diabetes','sex'],axis=1) 
x = x.to_numpy()
#print(x.shape)   # (442, 9)   ---> (442, 3, 3, 1)

x = x.reshape(x.shape[0], 3, 3, 1)  
#print(x.shape)   # (442, 3, 3, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), padding='same', strides=1, input_shape=(3, 3, 1)))  
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(MaxPooling2D())  
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'                
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   
model_path = "".join([filepath, 'k35_diabetes_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, callbacks=[es, mcp])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
<컬럼 1개 삭제했을때>
loss : 3677.759521484375
r2스코어 :  0.4333225606956018
<MaxPooling2D 함께 적용 후>
loss : 4169.42919921875
r2스코어 :  0.357564930177768

=> column을 drop시키지않고 실행했을때 가장 효과가 좋음 
'''
