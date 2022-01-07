import pandas as pd
from pandas.core.frame import DataFrame
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import pickle, csv
import numpy as np
from pandas import Series, DataFrame

# [ 1: 관리종목 , 0: 안전종목 ] - 유가증권
# 관리종목 18개, 안전종목 18개
   
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
    dataset.columns = ['유동비율1','당좌비율1','부채비율1','유보율1','순차입금비율1','이자보상배율1','매출액증가율1','판매비와관리비증가율1','EBITDA증가율1','매출총이익율1','영업이익률1',
                       '유동비율2','당좌비율2','부채비율2','유보율2','순차입금비율2','이자보상배율2','매출액증가율2','판매비와관리비증가율2','EBITDA증가율2','매출총이익율2','영업이익률2',
                       '유동비율3','당좌비율3','부채비율3','유보율3','순차입금비율3','이자보상배율3','매출액증가율3','판매비와관리비증가율3','EBITDA증가율3','매출총이익율3','영업이익률3',
                       '유동비율4','당좌비율4','부채비율4','유보율4','순차입금비율4','이자보상배율4','매출액증가율4','판매비와관리비증가율4','EBITDA증가율4','매출총이익율4','영업이익률4',
                       '유동비율5','당좌비율5','부채비율5','유보율5','순차입금비율5','이자보상배율5','매출액증가율5','판매비와관리비증가율5','EBITDA증가율5','매출총이익율5','영업이익률5']
    # print(dataset)
    
    # print(dataframe.shape)
    dataset_all = dataset_all.append(dataset)
    

dataset_all["Target"] = 1
print(dataset_all)  

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