import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터
path = "../_data/kaggle/bike/"  # '.'은 현재 나의 작업폴더, '..'은 현재폴더의 전 폴더

train = pd.read_csv(path + 'train.csv')
#print(train)  
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file)  
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file)  
#print(submit_file.shape)  # (6493, 2)
#print(submit_file.columns)  # ['datetime', 'count']

#print(type(train))  # <class 'pandas.core.frame.DataFrame'>
#print(train.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    10886 non-null  object   -> 'object'는 상위자료형; print(train.info())에서 object가 나오면 "스트링string(문자열)'으로 생각하면 됨
 1   season      10886 non-null  int64    
 2   holiday     10886 non-null  int64
 3   workingday  10886 non-null  int64
 4   weather     10886 non-null  int64
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64
 10  registered  10886 non-null  int64
 11  count       10886 non-null  int64
dtypes: float64(3), int64(8), object(1)
memory usage: 1020.7+ KB
None
'''
#print(train.describe())
''' # count: 총 갯수, mean: 평균, std: 표준편차
             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000
'''
#print(train.columns)
#Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#      'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#     dtype='object')
#print(train.head())
#print(train.tail())

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  # 컬럼을 삭제할때는 axis=1, 디폴트값은 axis=0
test_file = test_file.drop(['datetime'], axis=1)  

#print(x.columns)
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed'],
#      dtype='object')
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y)
#print(y.shape)  # (10886,)

# 로그변환
y = np.log1p(y)  # log1p : y값을 log변환하기 전에 1을 더해주는 함수

#plt.plot(y)
#plt.show()

# 데이터가 많거나 한쪽으로 치우쳐진(쏠린) 경우 log를 씌워준다. y=0 또는 x=0가 나타나는 경우, 로그 변환은 가능하지 않으므로 log변환 하기 전에 1을 더해주고 계산.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8)) 
model.add(Dense(100, activation='linear')) 
model.add(Dense(50)) 
model.add(Dense(20)) 
model.add(Dense(10)) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)

rmse = RMSE(y_test, y_pred)
print("RMSE: ", rmse)


#로그변환 전
#loss : 24370.427734375
#r2스코어 :  0.22897276615165385
#RMSE:  156.11028763767933

#로그변환 후
#loss : 1.4604376554489136
#r2스코어 :  0.2547093149477536
#RMSE:  1.208485696261775


############ 제출용 제작 ##############
results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + "final.csv", index=False)

# 루트와 로그 (RMSE-MSE에 루트, RMSLE-로그)


# [과제] 중위값과 평균값의 차이를 찾아라 (비교,분석)
# 중앙값(median)은 자료 값의 크기 순서대로 나열한 후 가장 중앙에 위치하는 값.
# 평균(mean)은 자료의 값를 모두 더한 후에 그 전체자료 갯수로 나눈 값.
