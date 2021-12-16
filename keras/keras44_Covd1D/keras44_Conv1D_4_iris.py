import numpy as np, time
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    # (150, 4) (150,)
#print(np.unique(y))   # [0 1 2]
y = to_categorical(y)
# print(y.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0],x_train.shape[1],1)
x_test = scaler.fit_transform(x_test).reshape(x_test.shape[0],x_test.shape[1],1)
#print(x_train.shape, x_test.shape)    # (120, 4, 1) (30, 4, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(4,1)))
model.add(Flatten())
model.add(Dropout(0.2))               
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(3, activation='softmax'))
#model.summary()   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=False)

start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=[es])
end = time.time() - start

print("걸린시간: ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss[0])
print('accuracy :',loss[1])

y_predict = model.predict(x_test)

'''
걸린시간:  8.191 초
loss : 0.014783612452447414
accuracy : 0.9666666388511658

=> LSTM과 비교했을때 속도향상
'''