import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm  

#1. 데이터
path = "../_data/dacon/open/"    

train = pd.read_csv(path + 'train_data.csv')
print(train.shape)  # (24998, 4)
test = pd.read_csv(path + 'test_data.csv')
print(test.shape)  # (1666, 4)
submission = pd.read_csv(path + 'sample_submission.csv')#, dtype=float
print(submission.shape)  # (1666, 2)

feature = train['label']

plt.figure(figsize=(10,7.5)) # 그래프 이미지 크기 설정

plt.title('label', fontsize=20)
temp = feature.value_counts() # feature 변수의 변수별 개수 계산
plt.bar(temp.keys(), temp.values, width=0.5)
plt.xticks(temp.keys(), fontsize=15) 
plt.show()

