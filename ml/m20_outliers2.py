import  numpy as np
import pandas as pd
aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]])
# (2, 13) -> (13, 2)
aaa = np.transpose(aaa)   # (13, 2)
# print(aaa)
aaa = pd.DataFrame(aaa)
aaa.columns = ['a', 'b']


# def outliers(data_out, column):
#     quartile_1, q2, quartile_3 = np.quantile(data_out[column], 0.25), np.quantile(data_out[column], 0.5), np.quantile(data_out[column], 0.75)   
#     print("1사분위 : ", quartile_1)
#     print("q2 : ", q2)
#     print("3사분위 : ", quartile_3)
#     iqr = quartile_3 - quartile_1
#     print("iqr : ", iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     data1 = data_out[data_out[column] > upper_bound]     
#     data2 = data_out[data_out[column] < lower_bound]  
#     return print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.')   

# print(outliers(aaa,'a'))
# outliers_loc = outliers(aaa,'a')
# print("이상치의 위치 : ", outliers_loc)

# # import matplotlib.pyplot as plt
# # plt.boxplot(aaa)
# # plt.show()

'''
def outlier_iqr(data, column): 

    # lower, upper 글로벌 변수 선언하기     
    global lower, upper    
    
    # 4분위수 기준 지정하기  

    q25, q75 = np.quantile(data_out[column], 0.25), np.quantile(data_out[column], 0.75)          
    
    # IQR 계산하기     
    iqr = q75 - q25    
    
    # outlier cutoff 계산하기     
    cut_off = iqr * 1.5          
    
    # lower와 upper bound 값 구하기     
    lower, upper = q25 - cut_off, q75 + cut_off     
    
    print('IQR은',iqr, '이다.')     
    print('lower bound 값은', lower, '이다.')     
    print('upper bound 값은', upper, '이다.')    
    
    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기     
    data1 = data_out[data_out[column] > upper]     
    data2 = data_out[data_out[column] < lower]    
    
    # 이상치 총 개수 구하기
    return print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.')

outliers_loc = outliers(aaa, 1)
print("이상치의 위치 : ", outliers_loc)
'''