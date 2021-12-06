from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn import datasets
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()    


#model.save("./_save/keras25_1_save_model.h5")  
#model.save_weights("./_save/keras25_1_save_weights.h5")  


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    # ModelCheckpoint는 EarlyStopping과 함께 쓰는 것이 효과적임
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 
                   #restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌


start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp])

end = time.time() - start

print("======================================")
print(hist)
print("======================================")
print(hist.history)
print("======================================")
print(hist.history['loss'])
print("======================================")
print(hist.history['val_loss'])
print("======================================")


import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()


print("걸린시간: ", round(end, 3),'초')

model.save("./_save/keras26_1_save_model.h5")  


#model.save("./_save/keras25_3_save_model.h5")  
#model.save_weights("./_save/keras25_3_save_weights.h5")  

#model.load_weights('./_save/keras25_1_save_weights.h5')
#model.load_weights('./_save/keras25_3_save_weights.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#loss : 27.93290901184082
#r2스코어 :  0.6658061928158581


##################################################  [ N O T E ]  #################################################################
'''
콜백함수 : 특정조건에서 자동으로 실행되는 함수
keras의 콜백함수 2가지 -> EarlyStopping, ModelCheckpoint

ModelCheckpoint : 학습중인 모델 자동으로 저장하기
monitor에 val_loss를 지정해주는건 모델개선여부를 val_loss로 모니터하겠다는 의미
save_best_only 에 True를 지정하면 모델이 이전에 비해 개선되었을때만 저장. False로 지정하면 모든 epochs마다 저장
verbose=1 로 지정하면 실행되면서 개선되는 것을 볼수 있음

=> ModelCheckpoint를 사용함으로써 모델이 과적합이 진행되더라도 과적합이 진행되기 전 모델을 불러와 사용할 수 있음
EarlyStopping이 끝나기 전까지의 최소의 로스값이 적용된 최적의 가중치가 저장됨



EarlyStopping
restore_best_weights=True 로 지정하면 model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원함. False라면 마지막 training이 끝난 후의 weight로 나둠


'''