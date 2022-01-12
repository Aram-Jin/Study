import pandas as pd
import numpy as np
import requests
from pandas.core.frame import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")
data3 = pd.read_csv("./평가종목data.csv")

# print(type(data1))
# print(type(data2))

dataset = pd.concat([data1,data2],ignore_index=True).astype('float')
pre_data = data3.astype('float')
# print(type(pre_data))
del dataset['Unnamed: 0']
del pre_data['Unnamed: 0']

# print(type(dataset))
# print(type(pre_data))
# print(dataset.info())
# print(pre_data.info())

# print(dataset)

# dataset.to_csv('data_reset.csv', index=True, encoding='utf-8-sig')

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

# result_data = DecisionTreeClassifier(random_state=100)
# result_skfold = StratifiedKFold(n_splits=2)
# idx_iter=0
# cv_accuracy=[]

# for train_index, test_index in result_skfold.split(features, label):
#     x_train, x_test = features[train_index], features[test_index]
#     y_train, y_test = label[]
    
    
# model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)   







# print(np.unique(y))    # [0 1] 

# print(x.shape)   # (36, 55)
# print(y.shape)   # (36,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=55))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 


# 교차검증

kfold = KFold(n_splits=2, shuffle=True, random_state=49) 
K_cross_val_score = cross_val_score(model, x, y, cv=kfold)

print(K_cross_val_score)

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

# model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) 


# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ',loss[0])
# print('accuracy : ',loss[1])

# resulte = model.predict(pre_data)
# print(resulte)
# print(type(resulte))
# print(resulte.shape)





# # {'safe': 0, 'danger': 1}

# # for print_result()

# # if(resulte[0][0]<=0.5):
# #     safe = 100 - resulte[0][0]*100
# #     print(f"이 종목은 {round(safe,2)} % 확률로 투자해도 안전한 종목입니다")
# # elif(resulte[0][0]>=0.5):
# #     danger = resulte[0][0]*100
# #     print(f"이 종목은 {round(danger,2)} % 확률로 관리종목으로 지정될 가능성이 있는 위험한 종목입니다")
# # else:
# #     print("ERROR")
    
