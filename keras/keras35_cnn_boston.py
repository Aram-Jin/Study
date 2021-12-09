from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import numpy as np, pandas as pd, datetime

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#print(x)  # 단위 구분 ex) 6.3200e-03 -> 'e-03' 은 소숫점 3자리(0.000x) 'e+03' 은 반대로 
#print(y)
#print(x.shape)  # (506, 13) -----> (506,13,1,1) or (506,1,1,13)
#print(y.shape)  # (506,)

#print(datasets.feature_names)
#print(datasets.DESCR)

xx = pd.DataFrame(x, columns=datasets.feature_names)
#print(type(xx))
#print(xx)
#print(xx.corr())   # --> 상관관계

xx['price'] = y 
#print(xx)
#print(xx.corr())   # --> 상관관계

#import matplotlib.pyplot as plt
#import seaborn as sns  # 좀더 예쁘게 만들어주는
#plt.figure(figsize=(10,10))
#sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)  # cbar=컬러바
#plt.show()

x = xx.drop(['CHAS', 'price'],axis=1) 
x = x.to_numpy()
#print(x.shape)   # (506, 12)

x = x.reshape(x.shape[0], 3, 2, 2)  
#print(x.shape, y.shape)  # (506, 12, 1, 1) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), padding='same', strides=1, input_shape=(3, 2, 2)))  
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
model_path = "".join([filepath, 'k35_boston_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=model_path)   

model.fit(x_train, y_train, epochs=1000, batch_size=40, verbose=1, validation_split=0.2, callbacks=[es, mcp]) 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


'''
loss : 21.135908126831055
r2스코어 :  0.7471265619876881
'''
