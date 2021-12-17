#1. 데이터                                                            # 앙상블기법은 행의 갯수가 같아야 사용가능!!
import numpy as np

x1 = np.array([range(100), range(301, 401)])     # 삼성 저가, 고가
x1 = np.transpose(x1)

y1 = np.array(range(1001, 1101))   # 삼성전자 종가
y2 = np.array(range(101, 201))     # 하이닉스 종가
y3 = np.array(range(401, 501))     # 하이닉스 종가


print(x1.shape, y1.shape, y2.shape, y3.shape,)    # (100, 2) (100,) (100,) (100,)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1, y1, y2, y3, train_size=0.8, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
print(y1_train.shape, y1_test.shape)    # (80,) (20,)
print(y2_train.shape, y2_test.shape)    # (80,) (20,)
print(y3_train.shape, y3_test.shape)    # (80,) (20,)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

#2-1 모델1
input1 = Input(shape=(2,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

from tensorflow.keras.layers import concatenate, Concatenate      #concatenate -> 노드 두개를 순서대로 결합하여 하나의 노드로 만듬

#2-2 output모델1
output21 = Dense(7)(output1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-3 output모델2
output31 = Dense(7)(output1)
output32 = Dense(11)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

#2-4 output모델3
output41 = Dense(7)(output1)
output42 = Dense(11)(output41)
output43 = Dense(21)(output42)
output44 = Dense(11, activation='relu')(output43)
last_output3 = Dense(1)(output44)

model = Model(inputs=input1, outputs=[last_output1,last_output2, last_output3])
model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam', metrics=['mae'])  # metrics는 훈련에 영향을 미치지 않지만 평가지표로 사용가능

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x1_train, [y1_train,y2_train, y3_train], epochs=500, verbose=1, validation_split=0.2, callbacks=[es]) 


#4. 평가, 예측
results = model.evaluate(x1_test, [y1_test, y2_test, y2_test])
print('loss : ',results[0])   # loss :  92370.421875
print(results)     # [92370.421875, 68.38665771484375, 0.06699138879776001, 92301.96875, 6.992388725280762, 0.16098061203956604, 303.5365905761719]

y_predict = model.predict(x1_test)
# print(y_predict)

# from sklearn.metrics import r2_score
# r2 = r2_score([y1_test, y2_test, y3_test], y_predict)
# print('r2스코어 : ', r2)

