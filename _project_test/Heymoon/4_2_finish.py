import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import pickle, csv

codes = ['001520','001740','009270','004560','016380','010580','013000','019490','144620','015540','002630','012600','003620','011300','011690','096760','234080','002420']

def fs_data(code):
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    res = requests.get(url)
    df = pd.read_html(res.text)
    return df[0]

for code in codes:
    # print(tabulate(fs_data(code), headers='keys', tablefmt='psql'))
    dataframe = fs_data(code)
    dataframe = dataframe.set_index(dataframe.columns[0])
    dataframe = dataframe[dataframe.columns[:]]
    dataframe = dataframe.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기','유보율계산에 참여한 계정 펼치기',
                           '순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기',
                           '판매비와관리비증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','매출총이익율계산에 참여한 계정 펼치기',
                           '영업이익률계산에 참여한 계정 펼치기']]
    # print(tabulate(dataframe,headers='keys',tablefmt='psql'))
    dataframe = dataframe.T
    dataframe= pd.DataFrame(dataframe)
    print(dataframe)
    
# print(type(dataframe))  # <class 'pandas.core.frame.DataFrame'>
# print(dataframe.shape)  # (5, 11)

# dataframe.to_csv('text_night.csv', header=False, index=False, encoding='utf-8-sig')   