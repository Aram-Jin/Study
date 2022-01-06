import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

def fs_data(temp_df):

    temp_df = temp_df.set_index(temp_df.columns[0])
    temp_df = temp_df[temp_df.columns[:]]
    temp_df = temp_df.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기','유보율계산에 참여한 계정 펼치기',
                           '순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기',
                           '판매비와관리비증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','매출총이익율계산에 참여한 계정 펼치기',
                           '영업이익률계산에 참여한 계정 펼치기']]
    temp_df = temp_df.T
    return temp_df

codes = ['001520', '001740','009270','004560','016380','010580','002630','013000','019490','144620','015540']

for code in codes:
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    # print(url)
    res = requests.get(url)
    df = pd.read_html(res.text)
    data_frame = df[0]
    fs_data_frame = fs_data(data_frame)
    
    # fs_data_frame.to_csv('{codes}_dataset.csv', header=True, index=True, encoding='utf-8-sig')
    
    print(fs_data_frame)
    
    # for num, code in enumerate(code):
    #     fs_data_frame = fs_data(code)
        
# fs_data_frame = fs_data
    
# fs_data_frame.to_csv('test_dataset.csv', header=True, index=True, encoding='utf-8-sig')
