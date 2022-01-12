import pandas as pd
import numpy as np
import requests, csv
from pandas.core.frame import DataFrame
from tabulate import tabulate
from bs4 import BeautifulSoup

# {1: 관리종목 , 0: 안전종목} - 유가증권
   
#1-1) 관리종목 데이터 생성
codes = ['001520','001740','009270','004560','016380','010580','013000','019490','144620','015540','002630','012600','003620','011300','011690','096760','234080','002420']

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

#2) 행열변환 및 테이블 수정 
dataset_all = pd.DataFrame()
for code in codes:
    dataframe = fs_data(code)
    dataframe = dataframe.T
    # dataframe.columns = ['유동비율','당좌비율','부채비율','유보율',
    #                        '순차입금비율','이자보상배율','매출액증가율',
    #                        '판매비와관리비증가율','EBITDA증가율','매출총이익율',
    #                        '영업이익률']
    # print(dataframe.shape)
   
    # dataframe = dataframe.reset_index(drop=False)
    dataframe_vl = dataframe.values
    datalist =[]
    datalist.append(dataframe_vl)
    data_np = np.array(datalist)
    # print(type(data_np))
    dataset = data_np.reshape(1,55)
    # print(dataset)
    dataset = pd.DataFrame(dataset)
    # print(dataset)
    dataset.columns = ['CR1','QR1','DR1','RR1','NDR1','ICR1','SGR1','SAEGR1','EBITDA1','GPM1','OPP1',
                       'CR2','QR2','DR2','RR2','NDR2','ICR2','SGR2','SAEGR2','EBITDA2','GPM2','OPP2',
                       'CR3','QR3','DR3','RR3','NDR3','ICR3','SGR3','SAEGR3','EBITDA3','GPM3','OPP3',
                       'CR4','QR4','DR4','RR4','NDR4','ICR4','SGR4','SAEGR4','EBITDA4','GPM4','OPP4',
                       'CR5','QR5','DR5','RR5','NDR5','ICR5','SGR5','SAEGR5','EBITDA5','GPM5','OPP5']
    # print(dataset)
    
    # print(dataframe.shape)
    dataset_all = dataset_all.append(dataset)

'''
유동비율 : CR
당좌비율 : QR
부채비율 : DR
유보율 : RR
순차입금비율 : NDR
이자보상배율 : ICR
매출액증가율 : SGR
판매비와관리비증가율 : SAEGR
EBITDA증가율 : EBITDA
매출총이익율 : GPM
영업이익률 : OPP
'''    
dataset_all = dataset_all.where(pd.notnull(dataset_all), '0')
dataset_all["Target"] = 1
# del dataset_all['Unnamed: 0']
print(dataset_all)  

print(dataset_all.info())  

'''
[18 rows x 56 columns]
<class 'pandas.core.frame.DataFrame'>
Int64Index: 18 entries, 0 to 0
Data columns (total 56 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   CR1      18 non-null     object
 1   QR1      18 non-null     object
 2   DR1      18 non-null     object
 3   RR1      18 non-null     object
 4   NDR1     18 non-null     object
 5   ICR1     18 non-null     object
 6   SGR1     18 non-null     object
 7   SAEGR1   18 non-null     object
 8   EBITDA1  18 non-null     object
 9   GPM1     18 non-null     object
 10  OPP1     18 non-null     object
 11  CR2      18 non-null     object
 12  QR2      18 non-null     object
 13  DR2      18 non-null     object
 14  RR2      18 non-null     object
 15  NDR2     18 non-null     object
 16  ICR2     18 non-null     object
 17  SGR2     18 non-null     object
 18  SAEGR2   18 non-null     object
 19  EBITDA2  18 non-null     object
 20  GPM2     18 non-null     object
 21  OPP2     18 non-null     object
 22  CR3      18 non-null     object
 23  QR3      18 non-null     object
 24  DR3      18 non-null     object
 25  RR3      18 non-null     object
 26  NDR3     18 non-null     object
 27  ICR3     18 non-null     object
 28  SGR3     18 non-null     object
 29  SAEGR3   18 non-null     object
 30  EBITDA3  18 non-null     object
 31  GPM3     18 non-null     object
 32  OPP3     18 non-null     object
 33  CR4      18 non-null     object
 34  QR4      18 non-null     object
 35  DR4      18 non-null     object
 36  RR4      18 non-null     object
 37  NDR4     18 non-null     object
 38  ICR4     18 non-null     object
 39  SGR4     18 non-null     object
 40  SAEGR4   18 non-null     object
 41  EBITDA4  18 non-null     object
 42  GPM4     18 non-null     object
 43  OPP4     18 non-null     object
 44  CR5      18 non-null     object
 45  QR5      18 non-null     object
 46  DR5      18 non-null     object
 47  RR5      18 non-null     object
 48  NDR5     18 non-null     object
 49  ICR5     18 non-null     object
 50  SGR5     18 non-null     object
 51  SAEGR5   18 non-null     object
 52  EBITDA5  18 non-null     object
 53  GPM5     18 non-null     object
 54  OPP5     18 non-null     object
 55  Target   18 non-null     int64
dtypes: int64(1), object(55)
memory usage: 8.0+ KB
None
'''

dataset_all.to_csv('관리종목data.csv', index=True, encoding='utf-8-sig')





#1-2) 안전종목 데이터 생성 (시총 상위 18종목)
# codes2 = ['005930', '000660', '207940', '035420', '051910', '005380', '035720', '006400', '000270', '005490', '323410', '012330', '068270', '105560', '096770', '066570', '028260', '377300']

# def fs_data2(code2):
#     url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code2}'
#     res = requests.get(url)
#     df = pd.read_html(res.text)
#     temp_df2 = df[0]
#     temp_df2 = temp_df2.set_index(temp_df2.columns[0])
#     temp_df2 = temp_df2[temp_df2.columns[:12]]
#     temp_df2 = temp_df2.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기','유보율계산에 참여한 계정 펼치기',
#                            '순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기',
#                            '판매비와관리비증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','매출총이익율계산에 참여한 계정 펼치기',
#                            '영업이익률계산에 참여한 계정 펼치기']]
#     return temp_df2

# dataset_all2 = pd.DataFrame()
# for code2 in codes2:
#     dataframe2 = fs_data(code2)
#     dataframe2 = dataframe2.T
#     dataframe2 = dataframe2.reset_index(drop=True)
#     dataframe2 = dataframe2.values
#     datalist2 =[]
#     datalist2.append(dataframe2)
#     data_np2 = np.array(datalist2)
#     dataset2 = data_np2.reshape(1,55)
#     dataset2 = pd.DataFrame(dataset2)
#     dataset_all2 = dataset_all.append(dataset2)

# dataset_all2["Target"] = 0

# # print(dataset_all)  
# print(dataset_all2)  
# dataset_all2.to_csv('안전종목data.csv', index=True, encoding='utf-8-sig')  
# # print(type(all))   
    # dataframe_np = np.array(dataframe)
    # dataframe_f = dataframe_np.shape(1,55)
    # print(tabulate(dataframe,headers='keys',tablefmt='psql'))
    
    # dataframe= pd.DataFrame(dataframe)
    # print(dataframe)
    
# print(type(dataset_all))  # <class 'pandas.core.frame.DataFrame'>
# print(dataset_all.shape)  # (1, 55)
# print(dataset_all)