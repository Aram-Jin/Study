import os
import time
import json
import datetime
import numpy as np
import pandas as pd
from fbprophet import Prophet

dataset = pd.read_csv('./data/ai_contest/raw_data.csv')
print(dataset.shape)   # (1311802, 9)
# print(dataset.head())
dataset = dataset.drop('Unnamed: 0', axis=1)
print(dataset.head())
print(np.unique(dataset['local']))

print(dataset.groupby('local')['local'].count())
# local
# (주)팜스토리엘피씨          2801
# SKC(수원)             2774
# ㈜코오롱워터앤에너지 경산사업소    2831
# ㈜코오롱워터앤에너지 울진사업소    2620
# 가평북면하수              1237
#                     ...
# 화천사창하수              1170
# 화천산양하수              1170
# 화천하수                1101
# 횡성하수                2821
# 효성(울산)              2831
# Name: local, Length: 545, dtype: int64
print(len(dataset['local'].value_counts()))  # 545





