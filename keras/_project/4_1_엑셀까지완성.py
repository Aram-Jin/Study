import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate


#1) 데이터 불러오기
def fs_data(code):
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
    res = requests.get(url)
    df = pd.read_html(res.text)
    temp_df = df[0]
    temp_df = temp_df.set_index(temp_df.columns[0])
    temp_df = temp_df[temp_df.columns[:]]
    temp_df = temp_df.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기',
                           '유보율계산에 참여한 계정 펼치기','순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기',
                           '자기자본비율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기','판매비와관리비증가율계산에 참여한 계정 펼치기',
                           '영업이익증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','EPS증가율계산에 참여한 계정 펼치기', 
                           '매출총이익율계산에 참여한 계정 펼치기','세전계속사업이익률계산에 참여한 계정 펼치기','영업이익률계산에 참여한 계정 펼치기', 
                           'EBITDA마진율계산에 참여한 계정 펼치기','ROA계산에 참여한 계정 펼치기','ROE계산에 참여한 계정 펼치기','ROIC계산에 참여한 계정 펼치기',
                           '총자산회전율계산에 참여한 계정 펼치기','총부채회전율계산에 참여한 계정 펼치기','총자본회전율계산에 참여한 계정 펼치기',
                           '순운전자본회전율계산에 참여한 계정 펼치기']]
    return temp_df

#2) 행열변환 및 테이블 수정    
def change_df(code, fs_df):
    for num,col in enumerate(fs_df.columns):
        temp_df = pd.DataFrame({code:fs_df[col]})
        temp_df = temp_df.T
        temp_df.columns = [[col]*len(fs_df), temp_df.columns]
        if num ==0:
            total_df = temp_df
        else:
            total_df = pd.merge(total_df, temp_df, how='outer', left_index=True, right_index=True)
    return total_df

#3) 본문코딩
code = ['001520','001740','009270','004560','016380','010580','002630','013000','019490','144620','015540']
for num, code in enumerate(code):
    fs_data_frame = fs_data(code)
    fs_data_frame_changed = change_df(code, fs_data_frame)
    if num ==0:
        total_fs = fs_data_frame_changed
    else:
        total_fs = pd.concat([total_fs, fs_data_frame_changed])
# print(total_fs)

    total_fs.to_csv('Financial.csv', header=True, index=True, encoding='utf-8-sig')

# for code in code:
#     # print(tabulate(fs_data(code), headers='keys', tablefmt='psql'))

#     dataframe = fs_data(code)
#     dataframe = dataframe.set_index(dataframe.columns[0])
#     dataframe = dataframe[dataframe.columns[:5]]
#     dataframe = dataframe.loc[['유동비율계산에 참여한 계정 펼치기','당좌비율계산에 참여한 계정 펼치기','부채비율계산에 참여한 계정 펼치기',
#                                '유보율계산에 참여한 계정 펼치기','순차입금비율계산에 참여한 계정 펼치기','이자보상배율계산에 참여한 계정 펼치기',
#                                '자기자본비율계산에 참여한 계정 펼치기','매출액증가율계산에 참여한 계정 펼치기','판매비와관리비증가율계산에 참여한 계정 펼치기',
#                                '영업이익증가율계산에 참여한 계정 펼치기','EBITDA증가율계산에 참여한 계정 펼치기','EPS증가율계산에 참여한 계정 펼치기',
#                                '매출총이익율계산에 참여한 계정 펼치기','세전계속사업이익률계산에 참여한 계정 펼치기','영업이익률계산에 참여한 계정 펼치기',
#                                'EBITDA마진율계산에 참여한 계정 펼치기','ROA계산에 참여한 계정 펼치기','ROE계산에 참여한 계정 펼치기','ROIC계산에 참여한 계정 펼치기',
#                                '총자산회전율계산에 참여한 계정 펼치기','총부채회전율계산에 참여한 계정 펼치기','총자본회전율계산에 참여한 계정 펼치기','순운전자본회전율계산에 참여한 계정 펼치기']]
    # print(tabulate(dataframe,headers='keys',tablefmt='psql'))
    
    
    