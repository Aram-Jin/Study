from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np, datetime

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)   # DESCR: 데이터셋에 대한 간략한 설명
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)   # (569, 30) (569,)
#print(np.unique(y))   # [0 1]

y = to_categorical(y)   
#print(y.shape)    # --> (569, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,3,2)
x_test = scaler.transform(x_test).reshape(len(x_test),5,3,2)


#2. 모델구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), padding='same', strides=1, input_shape=(5,3,2)))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  
filepath = './_ModelCheckPoint/'                
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    
model_path = "".join([filepath, 'k35_cancer_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])

y_predict = model.predict(x_test)


'''
loss:  0.06568750739097595
accuracy:  0.9824561476707458
'''
