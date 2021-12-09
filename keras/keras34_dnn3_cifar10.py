import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import datetime

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])  
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])   

#print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape, y_train.shape)  # (50000, 3072) (50000, 10)
#print(x_test.shape, y_test.shape)    # (10000, 3072) (10000, 10)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=3072, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  
filepath = './_ModelCheckPoint/'                 
filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
model_path = "".join([filepath, 'k34_cifar10_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌

model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, mcp])  


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

'''
loss :  1.5928930044174194
accuracy :  0.42969998717308044
'''