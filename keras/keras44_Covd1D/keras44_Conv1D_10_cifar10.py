from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np, time, datetime

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 10)

# scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
# print(x_train.shape, x_test.shape)   # (50000, 3072) (10000, 3072)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 3072, 1)
x_test = x_test.reshape(10000, 3072, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3072, 1)))  
model.add(Conv1D(5, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  
filepath = './_ModelCheckPoint/'                 
filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
model_path = "".join([filepath, 'k42_cifar10_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1,
                      save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌

start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, mcp]) 
end = time.time() - start


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

print("걸린시간: ", round(end, 3), '초')
