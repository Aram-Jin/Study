import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
# np.save('./_save_npy/keras47_5_train_x.npy',arr=xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy',arr=xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy',arr=xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy',arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')

print(x_train)
print(x_train.shape)   # (160, 150, 150, 3)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (150,150,3))) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=300, batch_size=10, verbose=1, validation_split=0.2, callbacks=[es])  

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

'''
loss :  0.08703574538230896
accuracy :  0.9833333492279053
'''