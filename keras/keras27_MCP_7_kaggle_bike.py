from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_predict))

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

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
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

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(8,))
dense1 = Dense(100)(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)
'''
model = Sequential()
model.add(Dense(100, input_dim=8)) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(50)) 
model.add(Dense(20)) 
model.add(Dense(10)) 
model.add(Dense(1)) 
'''
#model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 100)               900
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_4 (Dense)              (None, 10)                210
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 17,291
Trainable params: 17,291
Non-trainable params: 0
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
##################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k27_7_', datetime_spot, '_', filename])
####################################################################################################################################
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es, mcp])

model.save("./_save/keras27_7_save_kaggle_bike.h5")  


#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)


print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras27_7_save_kaggle_bike.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss :',loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)



############ 제출용 제작 ##############
results = model.predict(test_file)

submit_file['count'] = results

#print(submit_file[:10])
#submit_file.to_csv(path + "final.csv", index=False)


'''
Epoch 00177: val_loss did not improve from 1.26734
Epoch 00177: early stopping
================================== 1. 기본 출력  =========================================
69/69 [==============================] - 0s 443us/step - loss: 1.3537
loss : 1.3537006378173828
r2스코어 :  0.30917941071699573
RMSE :  1.1634863813297023
================================ 2. load_model 출력  =========================================
69/69 [==============================] - 0s 558us/step - loss: 1.3537
loss : 1.3537006378173828
r2스코어 :  0.30917941071699573
RMSE :  1.1634863813297023
'''
