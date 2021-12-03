import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
path = "../_data/dacon/wine/"
train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (3231, 14)

test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (3231, 13)

submit_file = pd.read_csv(path + 'sample_submission.csv')  
#print(submit_file.shape)  # (3231, 2)
#print(submit_file.columns)  # ['id', 'quality'], dtype='object

#print(type(train))  # <class 'pandas.core.frame.DataFrame'>
#print(train.info())
'''
Data columns (total 14 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   id                    3231 non-null   int64
 1   fixed acidity         3231 non-null   float64
 2   volatile acidity      3231 non-null   float64
 3   citric acid           3231 non-null   float64
 4   residual sugar        3231 non-null   float64
 5   chlorides             3231 non-null   float64
 6   free sulfur dioxide   3231 non-null   float64
 7   total sulfur dioxide  3231 non-null   float64
 8   density               3231 non-null   float64
 9   pH                    3231 non-null   float64
 10  sulphates             3231 non-null   float64
 11  alcohol               3231 non-null   float64
 12  type                  3231 non-null   object
 13  quality               3231 non-null   int64
dtypes: float64(11), int64(2), object(1)
memory usage: 353.5+ KB
None
'''
#print(train.describe())
'''
                id  fixed acidity  volatile acidity  citric acid  residual sugar  ...      density           pH    sulphates      alcohol      quality
count  3231.000000    3231.000000       3231.000000  3231.000000     3231.000000  ...  3231.000000  3231.000000  3231.000000  3231.000000  3231.000000
mean   1616.000000       7.205772          0.336072     0.319496        5.454813  ...     0.994667     3.214166     0.531455    10.497108     5.829155
std     932.853686       1.295494          0.160285     0.145854        4.816098  ...     0.003054     0.161873     0.149686     1.193813     0.850003
min       1.000000       3.800000          0.080000     0.000000        0.600000  ...     0.987110     2.720000     0.220000     8.400000     4.000000
25%     808.500000       6.400000          0.227500     0.250000        1.800000  ...     0.992205     3.100000     0.430000     9.500000     5.000000
50%    1616.000000       7.000000          0.290000     0.310000        3.100000  ...     0.994840     3.200000     0.510000    10.300000     6.000000
75%    2423.500000       7.700000          0.400000     0.390000        8.100000  ...     0.996900     3.320000     0.600000    11.300000     6.000000
max    3231.000000      15.900000          1.040000     1.660000       65.800000  ...     1.038980     4.010000     1.980000    14.900000     8.000000

[8 rows x 13 columns]
'''
#print(train.columns)
#Index(['id', 'fixed acidity', 'volatile acidity', 'citric acid',
#       'residual sugar', 'chlorides', 'free sulfur dioxide',
#       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type',
#       'quality'],
#      dtype='object')

x = train.drop(['id', 'quality'], axis=1) # 컬럼을 삭제할때는 axis=1, 디폴트값은 axis=0
test_file = test_file.drop(['id'], axis=1)
y = train['quality']

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

print("Label mapping: ")
for i, item in enumerate(le.classes_):
    print(item, '-->', i)

'''
Label mapping: 
red --> 0
white --> 1
'''   

le.fit(test_file['type']) 
test_file['type'] = le.transform(test_file['type'])

print(test_file['type'])  

print(np.unique(y))  # [4 5 6 7 8]

y = pd.get_dummies(y)  # pd.get_dummies 처리 : 결측값을 제외하고 0과 1로 구성된 더미값이 만들어진다. 
# 결측값 처리(dummy_na = True 옵션) : Nan을 생성하여 결측값도 인코딩하여 처리해준다.
# y = pd.get_dummies(y, drop_first=True) : N-1개의 열을 생성
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(12,))
dense1 = Dense(30)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(5, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)




############ 제출용 제작 ##############
results = model.predict(test_file)

submit_file['quality'] = results

print(submit_file[:10])

submit_file.to_csv(path + "final.csv", index=False)



'''
loss :  1.198427677154541
accuracy :  0.44667696952819824
'''


'''
loss :  1.0135687589645386
accuracy :  0.5641421675682068
'''

'''
loss :  1.0240126848220825
accuracy :  0.5610510110855103
'''



'''
loss :  0.9934438467025757
accuracy :  0.5656877756118774
'''