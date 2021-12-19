import numpy as np, pandas as pd, datetime, time
import matplotlib.pyplot as plt,  seaborn as sns  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping

def split_xy5(dataset, time_steps, y_column):
    x,y = list(), list()  # x와 y에 담겠다.
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    # 몇개로 나누어 줄것인가
        y_end_number = x_end_number + y_column    # y값 의미
        
        if y_end_number > len(dataset):   # 계속 반복하되 우리가 뽑으려는 결과값이 나오면 break하겠다.
            break
        
        tmp_x = dataset[i:x_end_number, :]  # 4개 행, 모든열
        tmp_y = dataset[x_end_number: y_end_number, 3]  # 3이니까 자릿값 0,1,2
        x.append(tmp_x)
        y.append(tmp_y)   
    return np.array(x),np.array(y)

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, header = 0, thousands =',', encoding='cp949')

samsung = samsung.iloc[:893,:].drop(['전일비','Unnamed: 6','등락률', '거래량','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1).sort_values(['일자'],ascending=[True]) 
kiwoom = kiwoom.iloc[:893,:].drop(['전일비','Unnamed: 6','등락률', '거래량','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1).sort_values(['일자'],ascending=[True])  

# samsung = samsung.loc[::-1]
# kiwoom = kiwoom.loc[::-1]

samsung = samsung.values
kiwoom = kiwoom.values

# high_price = samsung['고가'].values
# low_price = samsung['저가'].values
# mid_price = (high_price + low_price) / 2

# print(samsung.index, kiwoom.index)

# day_divided = 50
# day_length = day_divided + 1
# day_result = []
# for i in range(len(mid_price) - day_length):
#     day_result.append(mid_price[i: i + day_length])

print(samsung)    
print(kiwoom)


# x1 = samsung
# y1 = samsung['종가']

# x2 = kiwoom
# y2 = kiwoom['종가'] 

x1, y1 = split_xy5(samsung,5,1)
x2, y2 = split_xy5(kiwoom,5,1)

# print(x1.shape, y1.shape)  # (888, 5, 4) (888, 1)
# print(x2.shape, y1.shape)  # (888, 5, 4) (888, 1)

x1 = x1.reshape(888,-1)
x2 = x2.reshape(888,-1)

# print(x1.shape)
# print(x2.shape)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=66)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)   # (710, 20) (178, 20)
print(x2_train.shape, x2_test.shape)   # (710, 20) (178, 20)

scaler = MinMaxScaler()
#scaler = StandardScaler()
# scaler = RobustScaler()
#scaler = MaxAbsScaler()
x1_train = scaler.fit_transform(x1_train).reshape(len(x1_train),5,4)
x1_test = scaler.transform(x1_test).reshape(len(x1_test),5,4)

x2_train = scaler.fit_transform(x2_train).reshape(len(x2_train),5,4)
x2_test = scaler.transform(x2_test).reshape(len(x2_test),5,4)

print(x1_train.shape)   # (710, 5, 4)
print(x2_train.shape)   # (710, 5, 4)

#2. 모델구성

#2-1 모델1
input1 = Input(shape=(5,4))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

#2-1 모델2
input2 = Input(shape=(5,4))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output2 = Dense(7, activation='relu', name='output2')(dense3)

merge1 = Concatenate(axis=1)([output1, output2])

#2-3 output모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-4 output모델2
output31 = Dense(7)(merge1)
output32 = Dense(11)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])

# print(x_train.shape, x_test.shape, test_file.shape)   # (8708, 8, 1) (2178, 8, 1) (6493, 8, 1)

# def split_x(dataset, size):
#     list = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         list.append(subset)
#     return np.array(list)

# train = split_x(x1, 5)
# print(train)
# print(train.shape)  # (889, 5, 2)

# print(samsung.columns)
# print(kiwoom.columns)

# print(samsung)
# print(kiwoom)
# print(samsung.shape)  # (893, 2)
# print(kiwoom.shape)  # (893, 2)

# range(samsung.shape[0]-1,-1)
# print(samsung.head())
# range(samsung.shape[0]-1,-1,::-1)
# print(samsung.head())

# # print(samsung.shape)
# print(samsung.head())
# print(samsung.tail())
# print(kiwoom.head())
# print(kiwoom.tail())




# # print(type(samsung))  # <class 'pandas.core.frame.DataFrame'>
# # print(type(kiwoom))  # <class 'pandas.core.frame.DataFrame'>

# # print(samsung.info())
# # print(kiwoom.info())
# # print(samsung.describe())
# # print(kiwoom.describe())



# # x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, train_size=0.8, shuffle=True, random_state=66)




# # plt.figure(figsize=(10,10))
# # sns.heatmap(data=samsung.columns.corr(), square=True, annot=True, cbar=True) 
# # plt.show()

# # samsung.pngdf = samsung.loc[samsung['연도']>=1990]

# # plt.figure(figsize=(16, 9))
# # # sns.lineplot(y=df['종가'], x=df['일자'])
# # plt.xlabel('time')
# # plt.ylabel('price')

# # pd.to_datetime(samsung['일자'], format='%Y%m%d')
# # # 0      2020-01-07
# # # 1      2020-01-06
# # # 2      2020-01-03
# # # 3      2020-01-02
# # # 4      2019-12-30

# # samsung['일자'] = pd.to_datetime(samsung['일자'], format='%Y%m%d')
# # samsung['연도'] =samsung['일자'].dt.year
# # samsung['월'] =samsung['일자'].dt.month
# # samsung['일'] =samsung['일자'].dt.day
# # print(samsung.to_datetime)

# # x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
# # #print(x.shape)  # (10886, 8)

# # y = train['count']
# # #print(y.shape)  # (10886,)





# # test_file = test_file.drop(['datetime'], axis=1)  