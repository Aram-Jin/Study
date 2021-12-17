import numpy as np, datetime
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

#print(np.unique(y_train, return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(y.shape)   # (60000, 10)

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(LSTM(30, return_sequences=False, input_shape=(28, 28)))    
model.add(Dropout(0.2)) 
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1)) 
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  
filepath = './_ModelCheckPoint/'                 
filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
model_path = "".join([filepath, 'k42_9_', datetime_spot, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1,
                      save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌

model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, mcp]) 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

'''
loss :  0.46201324462890625
accuracy :  0.8352000117301941

'''