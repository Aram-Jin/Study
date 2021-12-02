from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
dense1 = Dense(50)(input1)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(3, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)
'''
model = Sequential()
model.add(Dense(50, activation='linear', input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
'''
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                250
_________________________________________________________________
dense_1 (Dense)              (None, 10)                510
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 33
=================================================================
Total params: 903
Trainable params: 903
Non-trainable params: 0
'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)

''' 
=========================================== Scaler만 적용 후 결과 비교 =============================================================
1. No Scaler
loss :  0.06996951997280121
accuracy :  0.9666666388511658

2. MinMaxScaler
loss :  0.09163820743560791
accuracy :  0.9666666388511658

3. StandardScaler
loss :  0.066742442548275
accuracy :  0.9666666388511658

☆ 4. RobustScaler -> ok
loss :  0.06954094022512436
accuracy :  1.0

5. MaxAbsScaler 
loss :  0.07157178968191147
accuracy :  0.9666666388511658

================================================ Activation = 'relu' 추가 적용 후 TEST================================================= 
1. No Scaler (dense_1에 activation='relu' 적용) 
loss :  0.08132674545049667
accuracy :  0.9666666388511658

2. MinMaxScaler (dense_1에 activation='relu' 적용) 
loss :  0.09428596496582031
accuracy :  0.9666666388511658

3. StandardScaler (dense_1에 activation='relu' 적용) 
loss :  0.10353484004735947
accuracy :  0.9333333373069763

☆ 4. RobustScaler (dense_1에 activation='relu' 적용) 
loss :  0.0653466060757637
accuracy :  0.9666666388511658

5. MaxAbsScaler (dense에 activation='relu' 적용) 
loss :  0.06833016127347946
accuracy :  0.9666666388511658


==> 결론 : "iris"데이터는 RobustScaler로 돌렸을때 가장 효과가 좋았으며, activation='relu' 적용했을때 loss값이 조금 개선됨. 
            but accuracy가 떨어져서 활성화함수relu를 사용을했을때의 개선됨을 잘모르겠음
            
=============================================== input_shape 모델로 튜닝 후 TEST =======================================================
            
< RobustScaler & dense_1에 activation='relu' 적용  >

loss :  0.06303315609693527
accuracy :  0.9666666388511658

==> 별 차이 없음, 아주 미세하게 개선됨

'''
