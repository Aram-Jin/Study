################################################
'''
< 데이터 스케일링(Data Scaling) 종류 > => scaler 완벽히 이해하기!! 완전 중요!!!!!!

1. MinMaxScaler(a,b)
   - 각 특성의 하한값을 a, 상한값을 b로 스케일링합니다. a=0, b=1일 경우 Normalization으로 표기할 때도 있습니다. 분류보다 회귀에 유용합니다.
   
2. StandardScaler
  - 각 특성의 평균을 0, 분산을 1로 스케일링합니다. 즉 데이터를 정규분포로 만듭니다. 
    하한값과 상한값이 존재하지 않을 수 있기에, 어떤 알고리즘에서는 문제가 있을 수 있습니다. 회귀보다 분류에 유용합니다.
   
3. RobustScaler   
  - 각 특성들의 중앙값을 0, IQR(제3사분위수-제1사분위수,Q3-Q1)을 1로 스케일링합니다. StandardScaler와 비슷하지만, 이상치의 영향을 최소화합니다.

4. MaxAbsScaler
  - 각 특성을 절대값이 0과 1사이가 되도록 스케일링합니다. 즉, 모든 값은 -1과 1사이로 표현되며, 데이터가 양수일 경우 MinMaxScaler와 같습니다.
'''
################################################
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)  # (506, 13)
print(y.shape)
#print(np.min(x), np.max(x))   # 0.0 711.0
#x = x/711.  # 부동소수점이라 .을 사용
#x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(6, input_dim=13)) 
model.add(Dense(11)) 
model.add(Dense(24)) 
model.add(Dense(5, activation='relu')) 
model.add(Dense(80)) 
model.add(Dense(100)) 
model.add(Dense(80)) 
model.add(Dense(50)) 
model.add(Dense(25, activation='relu')) 
model.add(Dense(12))  
model.add(Dense(5)) 
model.add(Dense(2)) 
model.add(Dense(1)) 

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 6)                 84
_________________________________________________________________
dense_1 (Dense)              (None, 11)                77
_________________________________________________________________
dense_2 (Dense)              (None, 24)                288
_________________________________________________________________
dense_3 (Dense)              (None, 50)                1250
_________________________________________________________________
dense_4 (Dense)              (None, 80)                4080
_________________________________________________________________
dense_5 (Dense)              (None, 100)               8100
_________________________________________________________________
dense_6 (Dense)              (None, 80)                8080
_________________________________________________________________
dense_7 (Dense)              (None, 50)                4050
_________________________________________________________________
dense_8 (Dense)              (None, 25)                1275
_________________________________________________________________
dense_9 (Dense)              (None, 12)                312
_________________________________________________________________
dense_10 (Dense)             (None, 5)                 65
_________________________________________________________________
dense_11 (Dense)             (None, 2)                 12
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 3
=================================================================
Total params: 27,676
Trainable params: 27,676
Non-trainable params: 0
_________________________________________________________________
'''


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.2, callbacks=[es])  


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



''' 
=========================================== Scaler만 적용 후 결과 비교 =============================================================
1. No Scaler
loss : 19.078636169433594
r2스코어 :  0.7690715760244011

2. MinMaxScaler
loss : 18.182775497436523
r2스코어 :  0.7799151123027355

3. StandardScaler
loss : 17.108516693115234
r2스코어 :  0.7929179426620022

4. RobustScaler
loss : 18.378992080688477
r2스코어 :  0.7775401028112389

5. MaxAbsScaler
loss : 16.873977661132812
r2스코어 :  0.7957568339084006

================================================ Activation = 'relu' 추가 적용 후 TEST================================================= 

1. No Scaler
  ### No Scaler (dense_3에 activation='relu' 적용)
loss : 14.947704315185547
r2스코어 :  0.8190725263683063

  ### No Scaler (dense_5에 activation='relu' 적용)
loss : 16.556838989257812
r2스코어 :  0.7995955095783518

☆ ### No Scaler (dense_3, 8 에 activation='relu' 적용) ; 파라미터갯수가 많은 레이어층에 앞뒤로 relu를 적용하니 파라미터갯수 줄어듬, " 아마도 최소의 loss값 "
loss : 13.04503345489502
r2스코어 :  0.8421024943386093

  ### No Scaler (dense_3,4,5,6,7,8 에 activation='relu' 적용) ; 파라미터갯수가 많은 레이어층 전체에 적용. 효과없는듯..
loss : 17.424734115600586
r2스코어 :  0.7890904677225382


2. MinMaxScaler (dense_3, 8 에 activation='relu' 적용)
loss : 10.91635513305664
r2스코어 :  0.8678680815199604

3. StandardScaler (dense_3, 8 에 activation='relu' 적용)
loss : 9.519649505615234
r2스코어 :  0.8847738593534034

4. RobustScaler (dense_3, 8 에 activation='relu' 적용)
loss : 10.325736999511719
r2스코어 :  0.8750169563680428

5. MaxAbsScaler (dense_3, 8 에 activation='relu' 적용)
loss : 13.996051788330078
r2스코어 :  0.8305913372178035


==>> 결론 : 아직 데이터 분석을 못해봤지만 activation='relu'를 적용해도 큰차이가 없는 것 같음

'''


##################################################  [ N O T E ]  #################################################################
'''
< 데이터 전처리 & 스케일링 >

numpy : 부동소수점 연산에 최적화 되어있는 라이브러리 (숫자로만 되어있기때문)
 - pandas의 경우, 숫자로 연산하기 힘든 여러가지 데이터가 있음 (object, string 등..) 

데이터(숫자)의 값이 크다면 부동소수점 연산하는 것이 효과적임
MinMaxScaler = ( ㅁ - Min ) / ( MaX - Min)

Data Scaling 시,
Scaling은 train만 하고 test, predict는 train비율에 적용하여 변환한다.
train의 범위만 생각하게되면 과적합 되기때문에 범위 밖의 데이터도 생각해주어야함
train(0~1), test, predict는 그 밖의 수가 나올수도있음
ex) scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

[참고]
https://ssoondata.tistory.com/25
https://m.blog.naver.com/wideeyed/221614354947
https://junklee.tistory.com/18
https://m.blog.naver.com/wideeyed/221293217463 -> scaler 비교

'''
