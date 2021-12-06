from tensorflow.keras.models import Sequential, load_model
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
#model = Sequential()
#model.add(Dense(40, input_dim=13))
#model.add(Dense(30))
#model.add(Dense(20))
#model.add(Dense(10))
#model.add(Dense(1))
#model.summary()    


#model.load_weights('./_save/keras25_1_save_weights.h5')
#model.load_weights('./_save/keras25_3_save_weights.h5')

#model.save("./_save/keras25_1_save_model.h5")  
#model.save_weights("./_save/keras25_1_save_weights.h5")  


#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    # ModelCheckpoint는 EarlyStopping과 함께 쓰는 것이 효과적임
#es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                      save_best_only=True, filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌


#start = time.time()
#hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp])

#end = time.time() - start

#print("======================================")
#print(hist.history['val_loss'])
#print("======================================")


#print("걸린시간: ", round(end, 3),'초')

#model.save("./_save/keras26_1_save_model.h5")  
model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')
#model = load_model('./_save/keras26_1_save_model.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#loss : 27.93290901184082
#r2스코어 :  0.6658061928158581

'''
loss : 45.525428771972656
r2스코어 :  0.45532622772158327
'''

'''
loss : 45.525428771972656
r2스코어 :  0.45532622772158327
'''

'''
loss : 45.525428771972656
r2스코어 :  0.45532622772158327
'''

'''

loss : 31.157007217407227
r2스코어 :  0.6272325125557762
'''

'''
loss : 31.157007217407227
r2스코어 :  0.6272325125557762
'''

'''
loss : 30.358333587646484
r2스코어 :  0.6367879161323043
'''



