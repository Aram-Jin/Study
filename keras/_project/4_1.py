import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

code = ['131100', '091970']

def fs_data(code):
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    res = requests.get(url)
    df = pd.read_html(res.text)
    return df[0]

for code in code:
    # print(tabulate(fs_data(code), headers='keys', tablefmt='psql'))

    dataframe = fs_data(code)
    dataframe = dataframe.set_index(dataframe.columns[0])
    dataframe = dataframe[dataframe.columns[:5]]
    dataframe = dataframe.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기',
                               '유보율계산에 참여한 계정 펼치기','순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기',
                               '자기자본비율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기','판매비와관리비증가율계산에 참여한 계정 펼치기',
                               '영업이익증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','EPS증가율계산에 참여한 계정 펼치기',
                               '매출총이익율계산에 참여한 계정 펼치기','세전계속사업이익률계산에 참여한 계정 펼치기','영업이익률계산에 참여한 계정 펼치기',
                               'EBITDA마진율계산에 참여한 계정 펼치기','ROA계산에 참여한 계정 펼치기','ROE계산에 참여한 계정 펼치기','ROIC계산에 참여한 계정 펼치기',
                               '총자산회전율계산에 참여한 계정 펼치기','총부채회전율계산에 참여한 계정 펼치기','총자본회전율계산에 참여한 계정 펼치기','순운전자본회전율계산에 참여한 계정 펼치기']]
    print(tabulate(dataframe,headers='keys',tablefmt='psql'))