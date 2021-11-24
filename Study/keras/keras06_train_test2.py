from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터                                # 파이썬 리스트 콜론, [ : ] 슬라이싱 ( List Slicing )
x = np.array([1,2,3,4,5,6,7,8,9,10])

y = np.array([1,2,3,4,5,6,7,8,9,10])

### 과제 ### Test
# train과 test비율을 8:2으로 분리하시오
x_train = x[:8]
x_test =  x[8:]
y_train = y[:8]
y_test =  y[8:]

'''
# 다른 방법
x_train = x[:8]
x_test =  x[-2:]
y_train = y[:8]
y_test =  y[-2:]
'''

print(x_train)
print(x_test)
print(y_train)
print(y_test)
