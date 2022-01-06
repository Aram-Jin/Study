import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import pickle, csv

codes = ['001520','001740','009270','004560','016380','010580','002630','013000','019490','144620','015540']

def fs_data(code):
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    res = requests.get(url)
    df = pd.read_html(res.text)
    return df[0]
# x=[]
for code in codes:
    # print(tabulate(fs_data(code), headers='keys', tablefmt='psql'))
    dataframe = fs_data(code)
    dataframe = dataframe.set_index(dataframe.columns[0])
    dataframe = dataframe[dataframe.columns[:]]
    dataframe = dataframe.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기',
                               '유보율계산에 참여한 계정 펼치기','순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기',
                               '자기자본비율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기','판매비와관리비증가율계산에 참여한 계정 펼치기',
                               '영업이익증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','EPS증가율계산에 참여한 계정 펼치기',
                               '매출총이익율계산에 참여한 계정 펼치기','세전계속사업이익률계산에 참여한 계정 펼치기','영업이익률계산에 참여한 계정 펼치기',
                               'EBITDA마진율계산에 참여한 계정 펼치기','ROA계산에 참여한 계정 펼치기','ROE계산에 참여한 계정 펼치기','ROIC계산에 참여한 계정 펼치기',
                               '총자산회전율계산에 참여한 계정 펼치기','총부채회전율계산에 참여한 계정 펼치기','총자본회전율계산에 참여한 계정 펼치기','순운전자본회전율계산에 참여한 계정 펼치기']]
    # print(tabulate(dataframe,headers='keys',tablefmt='psql'))
    
    dataframe= pd.DataFrame(dataframe)
    print(dataframe)
    for code in codes:
        dataframe.to_csv('{code}.csv', header=False, index=False, encoding='utf-8-sig')

    # x.append(dataframe)
   


    # dataframe.to_csv('dataframe.csv')
    # result = pd.DataFrame(list(tabulate(dataframe,headers='keys',tablefmt='psql')),columns=[])
    # print(type(result))
    # print(dataframe)
    # print(type(dataframe))
    
    # dataframe.to_pickle
    # print(type(result))
    
'''
    result = open('dataframe.pickle', 'wb')
    print(dataframe)
    pickle.dump(dataframe,result)
    result.close()
    
    result = open('dataframe.pickle', 'rb')
    dataframe = pickle.load(result)
    print(dataframe)
    
    print(type(dataframe))
    result.close()
''' 
    
    # dataframe.to_pickle("result.pkl")
    # dataframe = pd.read_pickle("result.pkl") 
    # print(type(dataframe))
# print(result) 
# result.to_csv('Financialtest.csv')   