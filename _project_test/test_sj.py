'''
from itertools import combinations
from sklearn.svm import OneClassSVM   #feature set에서 가장 성능이 좋은 조합 확인하는 방법

ocsvm=OneClassSVM( verbose=True, nu=0.00195889, kernel = 'rbf', gamma=0.0009)
 
def oc_model(model, x_train_df, x_test_df, y_test_df):
    model.fit(x_train_df)
    p=model.predict(x_test_df)
    cm = metrics.confusion_matrix(y_test_df, p)
    cm0=cm[0,0]
    cm1=cm[1,1]
    return cm0, cm1
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from tqdm import tqdm
# from tensorflow.keras.preprocessing import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup 
import requests
import csv
import time
import sys
from urllib.parse import urlencode
import xmltodict

#1) 데이터 로드하기

import requests
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote_plus, unquote
import json

# params = {'ServiceKey' : serviceKey,
#           'pageNo' : '1',
#           'numOfRows' : '10',
#           'startCreateDt' : '20200303', # 데이터 호출범위지정(시작일)
#           'endCreateDt' : '20211230' # 데이터 호출범위지정(종료일)
#          }

url = 'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701'
res = requests.get(url) # url 뒤에 params 정보를 붙임 
soup = BeautifulSoup(res.text, 'lxml')

# # xml -> 데이터프레임 전환
items = soup.find_all('item') # item의 하위 항목들만 모두 가져옵니다. items에 보관

# print(soup.head)
# # 최상단에 있는 item만 선택해서 컬럼명을 가져옴
# columns = []
# for item in items[0]:
#     columns.append(item.name) # item의 하위 항목 이름을 가져옵니다.


# 각 항목의 데이터 수집
final_data = []
for item in items:
    data = []
    for i in item: # item 하위 항목들을 한 줄씩 가져옵니다.

        # 21년 11월 24일 데이터 부터 accdefrate(누적 확진률) 데이터가 제외되므로 아예 수집에서 제외합니다.
        if i.name == 'accdefrate':
            pass
        else:
            data.append(i.text) # 항목에 해당하는 데이터를 가져옵니다.
    final_data.append(data)

# # 데이터프레임으로 전환
# df = pd.DataFrame(final_data, columns=columns)
# df

# # 영문 이름을 한글로 바꿉니다.
# df = df.rename(columns={
#     'seq': '게시글번호', #
#     'confcase': '확진자', 
#     'confcaserate': '확진률', 
#     'criticalrate': '치명률', 
#     'death': '사망자',   
#     'deathrate': '사망률',  
#     'gubun': '구분(성별, 연령별)', #
#     'createdt': '등록일시분초', #
#     'updatedt': '수정일시분초'#
# })
# # 원하는 순서대로 항목을 나열합니다.
# df = df[[
#     '게시글번호', '구분(성별, 연령별)', '확진자', '확진률', '치명률', '사망자','사망률', 
#     '등록일시분초', '수정일시분초'
# ]]

# # # 기준일을 추후 분석의 편의를 위해 날짜 타입으로 변경합니다.
# # df['기준일'] = pd.to_datetime(df['기준일'])
# print(df.info)
# # 엑셀 저장 (해당 .py또는 .ipynb 파일이 있는 폴더)
# df.to_csv("../개인프로젝트/keras1000_Corona(연령별,성별).csv", mode='w', encoding='euc-kr')
# df