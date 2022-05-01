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


dataset_all.to_csv('관리종목data.csv', index=True, encoding='utf-8-sig')

