import pandas as pd
from pandas.core.frame import DataFrame
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import pickle, csv
import numpy as np
from pandas import Series, DataFrame
# 1: 관리종목 , 0: 안전

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
    # print(tabulate(fs_data(code), headers='keys', tablefmt='psql'))
    dataframe = fs_data(code)
    dataframe = dataframe.T
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.values
    datalist =[]
    datalist.append(dataframe)
    data_np = np.array(datalist)
    dataset = data_np.reshape(1,55)
    dataset = pd.DataFrame(dataset)
    dataset_all = dataset_all.append(dataset)


# all = dataset_all 
dataset_all["Target"] = 1
# all = DataFrame.insert(column='관리종목',value=1,loc=-1,allow_duplicates = False)
print(dataset_all)    
# print(type(all))   
    # dataframe_np = np.array(dataframe)
    # dataframe_f = dataframe_np.shape(1,55)
    # print(tabulate(dataframe,headers='keys',tablefmt='psql'))
    
    # dataframe= pd.DataFrame(dataframe)
    # print(dataframe)
    
# print(type(dataset_all))  # <class 'pandas.core.frame.DataFrame'>
# print(dataset_all.shape)  # (1, 55)
# print(dataset_all)
dataset_all.to_csv('friyay_train.csv', index=True, encoding='utf-8-sig')