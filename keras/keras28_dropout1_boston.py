import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_boston
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dropout(0.2))                  
model.add(Dense(30))
#model.add(Dropout(0.3))   
model.add(Dense(20))
#model.add(Dropout(0.1))   
model.add(Dense(10))
model.add(Dense(1))
model.summary()   


#model.load_weights('./_save/keras25_1_save_weights.h5')
#model.load_weights('./_save/keras25_3_save_weights.h5')

#model.save("./_save/keras25_1_save_model.h5")  
#model.save_weights("./_save/keras25_1_save_weights.h5")  


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    # ModelCheckpoint는 EarlyStopping과 함께 쓰는 것이 효과적임
###################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k28_1_', datetime_spot, '_', filename])
####################################################################################################################################

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌


start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es,mcp])

end = time.time() - start

#print("걸린시간: ", round(end, 3),'초')

model.save("./_save/keras28_1_save_boston.h5")  
#model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')
#model = load_model('./_save/keras26_1_save_model.h5')


#4. 평가, 예측

print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras28_1_save_boston.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss :',loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)


'''
Epoch 00285: val_loss did not improve from 30.73880
Epoch 00285: early stopping
================================== 1. 기본 출력  =========================================
4/4 [==============================] - 0s 0s/step - loss: 18.4995
loss : 18.499534606933594
r2스코어 :  0.7786685475141868
================================ 2. load_model 출력  =========================================
4/4 [==============================] - 0s 0s/step - loss: 18.4995
loss : 18.499534606933594
r2스코어 :  0.7786685475141868

//////////////////////////////// dropout 적용후 결과 ///////////////////////////////////

Epoch 00352: val_loss did not improve from 32.04395
Epoch 00352: early stopping
================================== 1. 기본 출력  =========================================
4/4 [==============================] - 0s 997us/step - loss: 27.4159
loss : 27.41592025756836
r2스코어 :  0.6719914526092816
================================ 2. load_model 출력  =========================================
4/4 [==============================] - 0s 666us/step - loss: 27.4159
loss : 27.41592025756836
r2스코어 :  0.6719914526092816

'''


# 드랍아웃 적용시 와 미적용시 비교 성능향상 차이 비교