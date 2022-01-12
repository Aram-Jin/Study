import pandas as pd
from pandas.core.frame import DataFrame
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import pickle, csv
import numpy as np
from pandas import Series, DataFrame

# 1-2) 안전종목 데이터 생성 (시총 상위 120종목) -> 은행주, 우선주, SPAC 제외
codes = ['005930','000660','207940','035420','051910','006400','005380','035720','000270','068270','005490','012330','096770','066570','028260'
         ,'034730','051900','015760','009150','036570','003550','011200','017670','018260','034020','033780','010950','003670','010130','003490'
         ,'251270','090430','034220','011070','030200','011170','009830','009540','018880','086280','011790','004020','032640','097950','137310'
         ,'021240','035250','020150','011780','010140','000720','161390','028050','000100','241560','267250','271560','139480','003410','180640'
         ,'078930','006360','000990','307950','036460','002790','008930','002380','047810','128940','010620','272210','014680','007070','004990'
         ,'052690','028670','026960','008770','000120','204320','047050','012750','285130','042660','112610','030000','012450','047040','010060'
         ,'023530','006280','004170','001040','282330','064350','000880','093370','000080','011210','009240','111770','081660','012510','271940'
         ,'004370','004800','004000','000150','120110','006260','003090','042700','069960','001230','007310','010120','017800','069620','051600'
         ]

def fs_data(code):
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    res = requests.get(url)
    df = pd.read_html(res.text)
    temp_df = df[0]
    temp_df = temp_df.set_index(temp_df.columns[0])
    temp_df = temp_df[temp_df.columns[:6]]
    temp_df = temp_df.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기','유보율계산에 참여한 계정 펼치기',
                           '순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기',
                           '판매비와관리비증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','매출총이익율계산에 참여한 계정 펼치기',
                           '영업이익률계산에 참여한 계정 펼치기']]
    return temp_df

dataset_all = pd.DataFrame()
for code in codes:
    dataframe = fs_data(code)
    dataframe = dataframe.T
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.values
    datalist =[]
    datalist.append(dataframe)
    data_np = np.array(datalist)
    dataset = data_np.reshape(1,55)
    dataset = pd.DataFrame(dataset)
    dataset.columns = ['CR1','QR1','DR1','RR1','NDR1','ICR1','SGR1','SAEGR1','EBITDA1','GPM1','OPP1',
                       'CR2','QR2','DR2','RR2','NDR2','ICR2','SGR2','SAEGR2','EBITDA2','GPM2','OPP2',
                       'CR3','QR3','DR3','RR3','NDR3','ICR3','SGR3','SAEGR3','EBITDA3','GPM3','OPP3',
                       'CR4','QR4','DR4','RR4','NDR4','ICR4','SGR4','SAEGR4','EBITDA4','GPM4','OPP4',
                       'CR5','QR5','DR5','RR5','NDR5','ICR5','SGR5','SAEGR5','EBITDA5','GPM5','OPP5']
    dataset_all = dataset_all.append(dataset)

dataset_all = dataset_all.where(pd.notnull(dataset_all), '0')
dataset_all["Target"] = 0
print(dataset_all)  

dataset_all.to_csv('안전종목120data.csv', index=True, encoding='utf-8-sig')
