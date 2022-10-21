from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import os, re, json, tqdm
import pymysql
import openpyxl
import sys, csv
import datetime
import class3
import json

# 데이터 불러오기
PATH = 'C:/Users/GE66/AI/luvd_data/'

df = pd.read_csv(PATH + "2022-09-23Data_luvd.csv")#,  encoding='cp949', index_col=0)
types = pd.read_csv(PATH + "final_list_du_220916.csv",  encoding='cp949', index_col=0)


# df전처리(h1태그. 진단결과 추출)
df = df.dropna(subset=['plaintext']) # plaintext에 결측치 있는 행만 제거
types = types.drop(['Unnamed: 5', 'etc', 'counts'], axis='columns')

htags = df[df['plaintext'].str.contains("h1")]  # h1 태그가 속해있는 데이터 추출
htags_reset = htags.reset_index(drop=True)

y_data = htags_reset['plaintext']

result = []
for i , v in y_data.iteritems():
    soup = bs(v, "html.parser")
    bs_h1 = soup.select("h1 span")     
    strs = "".join(str(y_data.text) for y_data in bs_h1)
    result.append(strs.strip())
    
y = pd.DataFrame(result, columns=["result"])

df_result = pd.concat([y, htags_reset],axis=1)
dataframe = df_result.drop(['plaintext','Unnamed: 0'], axis='columns')

dataset = dataframe.replace(r'^\s*$', np.nan, regex=True)
dataset = dataset.dropna(subset=['result']) # result 결측치 있는 행만 제거

# print(dataset.head)
print(dataset.columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score,f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from transformers import *

train = dataset[['last', 'reason', 'action', 'try', 'reaction', 'valuable',
       'start', 'charmingLover', 'charmingCustomer', 'relation', 'event','result']]

train['last', 'reason', 'action', 'try', 'reaction', 'valuable',
       'start', 'charmingLover', 'charmingCustomer', 'relation', 'event'].fillna('NAN', inplace=True)

train['data'] = train['last']+train['reason']+train['action']+train['try']+train['reaction']+train['valuable']+train['start']+train['charmingLover']+train['charmingCustomer']+train['relation']+train['event']

