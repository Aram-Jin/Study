import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10], 
                    [2, 4, np.nan, 8, np.nan], 
                    [np.nan, 4, np.nan, 8, 10], 
                    [np.nan, 4, np.nan, 8, np.nan]])

print(data.shape)  # (4, 5)
data = data.transpose()
data.columns = ['a', 'b', 'c', 'd']
print(data)  
#       a    b     c    d
# 0   2.0  2.0   NaN  NaN
# 1   NaN  4.0   4.0  4.0
# 2   NaN  NaN   NaN  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제
# print(data.dropna())
# print(data.dropna(axis=0))   # axis=0은 행
# print(data.dropna(axis=1))   # axis=1은 열

#2. 특정값
# means = data.mean()  
# print(means)  # 평균값
# data = data.fillna(means)
# print(data)

#2. 특정값 - 중위값
meds = data.median()
print(meds)
data2 = data.fillna(meds)
print(data2)

#2-3. 특정값 - ffill, bfill : 첫번쩨데이터나 마지막 데이터가 결측치일 경우 fill method 경우에 따라 채워지지않음. 
data2 = data.fillna(method='ffill')
print(data2)
data2 = data.fillna(method='bfill')
print(data2)

# 연속으로 결측치가 있을 경우, limit을 주어 선택적으로 채울수 있음 
data2 = data.fillna(method='ffill', limit=1)
print(data2)
data2 = data.fillna(method='bfill', limit=1)
print(data2)

#2-3. 특정값채우기
data2 = data.fillna(747474)
print(data2)

###################################### 특정 컬럼만!! #####################################################

means = data['a'].mean()
print(means)
data['a'] = data['a'].fillna(means)
print(data)

meds = data['b'].median()
print(meds)
data['b'] = data['b'].fillna(meds)
print(data)
