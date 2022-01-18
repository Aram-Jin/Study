import pandas as pd
import numpy as np
import requests
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from itertools import combinations
from sklearn.svm import OneClassSVM

pre_code = ['000150','026960']

def fs_data(code):
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    res = requests.get(url)
    df = pd.read_html(res.text)
    temp_df = df[0]
    temp_df = temp_df.set_index(temp_df.columns[0])
    temp_df = temp_df[temp_df.columns[:12]]
    temp_df = temp_df.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기','유보율계산에 참여한 계정 펼치기',
                           '순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기',
                           '판매비와관리비증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','매출총이익율계산에 참여한 계정 펼치기',
                           '영업이익률계산에 참여한 계정 펼치기']]
    return temp_df

predict_all = pd.DataFrame()
for code in pre_code:
    dataframe = fs_data(code)
    dataframe = dataframe.T
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.values
    datalist =[]
    datalist.append(dataframe)
    data_np = np.array(datalist)
    predict_data = data_np.reshape(1,55)
    predict_data = pd.DataFrame(predict_data)
    predict_data.columns = ['CR1','QR1','DR1','RR1','NDR1','ICR1','SGR1','SAEGR1','EBITDA1','GPM1','OPP1',
                       'CR2','QR2','DR2','RR2','NDR2','ICR2','SGR2','SAEGR2','EBITDA2','GPM2','OPP2',
                       'CR3','QR3','DR3','RR3','NDR3','ICR3','SGR3','SAEGR3','EBITDA3','GPM3','OPP3',
                       'CR4','QR4','DR4','RR4','NDR4','ICR4','SGR4','SAEGR4','EBITDA4','GPM4','OPP4',
                       'CR5','QR5','DR5','RR5','NDR5','ICR5','SGR5','SAEGR5','EBITDA5','GPM5','OPP5']
    predict_all = predict_all.append(predict_data)
predict_all = predict_all.where(pd.notnull(predict_all), '0')

# print(predict_all)
# del predict_all['Unnamed: 0']
# print(type(predict_all))  # <class 'pandas.core.frame.DataFrame'> 

data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")
# print(type(data1))
# print(type(data2))

dataset = pd.concat([data1,data2],ignore_index=True)
del dataset['Unnamed: 0']
# dataset['col'].isnull()
# dataset.fillna(0)
# print(dataset)
# dataset.to_csv('friyayyy.csv', index=True, encoding='utf-8-sig')

# print(dataset.info())
# print(dataset.feature_names)
# print(dataset.DESCR)
# print(np.min(dataset), np.max(dataset))
# print(dataset.head)
# for col in dataset.columns:
#     print(col)
# print(dataset.index)
# np.array(dataset)

x = dataset.drop(['Target'], axis=1)
y = dataset['Target']
# print(np.unique(y))    # [0 1] 
# print(x)   # (36, 55)
# print(x.shape)   # (36, 55)
# print(y.shape)   # (36,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

# print(x.shape)
# print(x_test.shape)
# print(type(x_test))
# print(predict_all)
# print(type(predict_all))
# print(predict_all.shape)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=55))
model.add(Dense(48))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(18))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 


es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) 


# #4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ',loss[1])


# resulte = model.predict(predict_all)
print(predict_all)
# predict_all.reset()
# print(predict_all.class_indices)
# # {'safe': 0, 'danger': 1}
# if(predict_all[0][0]<=0.5):
#     safe = 100 - predict_all[0][0]*100
#     print(f"이 종목은 {round(safe,2)} % 확률로 투자해도 안전한 종목입니다")
# elif(predict_all[0][0]>=0.5):
#     danger = predict_all[0][0]*100
#     print(f"이 종목은 {round(danger,2)} % 확률로 관리종목으로 지정될 가능성이 있는 위험한 종목입니다")
# else:
#     print("ERROR")





# """
# ocsvm = OneClassSVM(verbose=True, nu=0.00195889, kernel='rbf', gamma=0.0009)

# def oc_model(model, x_train_df, x_test_df, y_test_df):
#     model.fit(x_train_df)
#     p = model.predict(x_test_df)
#     cm = metrics.confusion_metrics(y_test_df, p)
    
#     cm0=cm[0,0]
#     cm1 = cm[1,1]
#     return cm0, cm1
# """