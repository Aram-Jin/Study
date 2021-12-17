import numpy as np, pandas as pd, datetime, time
import matplotlib.pyplot as plt,  seaborn as sns  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, thousands =',', encoding='cp949')

samsung = samsung.iloc[:893,:].drop(['전일비','Unnamed: 6','등락률', '거래량','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)  
kiwoom = kiwoom.iloc[:893,:].drop(['전일비','Unnamed: 6','등락률', '거래량','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)  

samsung = samsung.loc[::-1]
kiwoom = kiwoom.loc[::-1]

x1 = samsung
y1 = samsung['종가']

# x2 = kiwoom
# y2 = kiwoom['종가'] 

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1,
                                                    train_size=0.8, shuffle=True, random_state=66)

def split_x(dataset, size):
    list = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        list.append(subset)
    return np.array(list)

train = split_x(x1, 5)
print(train)
print(train.shape)  # (889, 5, 2)

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



# # scaler = MinMaxScaler()
# #scaler = StandardScaler()
# # scaler = RobustScaler()
# #scaler = MaxAbsScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.fit_transform(x_test)
# # test_file = scaler.transform(test_file)

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