"""
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor
import autokeras as ak

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

path = '../_data/dacon/housing/'
datasets = pd.read_csv(path + 'train.csv', index_col=0, header=0)
print(datasets)
test_sets = pd.read_csv(path + 'test.csv', index_col=0, header=0)
submit_sets = pd.read_csv(path + 'sample_submission.csv', index_col=0, header=0)

print(datasets.info())

# Data columns (total 14 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   Overall Qual    1350 non-null   int64
#  1   Gr Liv Area     1350 non-null   int64
#  2   Exter Qual      1350 non-null   object   -> str으로 생각하면 됨(문자열)
#  3   Garage Cars     1350 non-null   int64
#  4   Garage Area     1350 non-null   int64
#  5   Kitchen Qual    1350 non-null   object
#  6   Total Bsmt SF   1350 non-null   int64
#  7   1st Flr SF      1350 non-null   int64
#  8   Bsmt Qual       1350 non-null   object
#  9   Full Bath       1350 non-null   int64
#  10  Year Built      1350 non-null   int64
#  11  Year Remod/Add  1350 non-null   int64
#  12  Garage Yr Blt   1350 non-null   int64
#  13  target          1350 non-null   int64
# dtypes: int64(11), object(3)
# memory usage: 158.2+ KB
# None

print(datasets.describe())
print(datasets.isnull().sum())    # null값 확인 ;없음

############################### 중복값 처리 ####################################
print("중복값 제거 전 : ", datasets.shape)
datasets = datasets.drop_duplicates()
print("중복값 제거 후 : ", datasets.shape)
# 중복값 제거 전 :  (1350, 14)
# 중복값 제거 후 :  (1349, 14)

############################## 이상치 확인 처리 #################################
outliers_loc = outliers(datasets['Garage Yr Blt'])
print(outliers_loc)
print(datasets.loc[[255], 'Garage Yr Blt'])   # 2207
datasets.drop(datasets[datasets['Garage Yr Blt']==2207].index, inplace=True)
print(datasets.shape)   # (1348, 14)


print(datasets['Exter Qual'].value_counts())
# TA    808
# Gd    485
# Ex     49
# Fa      8
print(datasets['Kitchen Qual'].value_counts())
# TA    660
# Gd    560
# Ex    107
# Fa     23
print(datasets['Bsmt Qual'].value_counts())
# TA    605
# Gd    582
# Ex    134
# Fa     28
# Po      1   -> datasets에서 Po를 삭제한다면 test데이터에서 Po값을 nan처리 해도 됨(부스팅계열에서 사용가능)

# print(test_sets['Exter Qual'].value_counts())
# print(test_sets['Kitchen Qual'].value_counts())   # Po 1
# print(test_sets['Bsmt Qual'].value_counts())   # Po  1

qual_cols = datasets.dtypes[datasets.dtypes == np.object].index
print(qual_cols)
# Index(['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'], dtype='object')

def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

datasets = label_encoder(datasets, qual_cols)
test_sets = label_encoder(test_sets, qual_cols)

print(datasets.shape)   # (1350, 14)
print(test_sets.shape)   # (1350, 13)

########################################## 분류형 컬럼을 one hot encoding #########################################
datasets = pd.get_dummies(datasets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test_sets = pd.get_dummies(test_sets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])

########################################## 분류형 컬럼을 one hot encoding #########################################
# print(datasets.columns)  
# print(test_sets.columns)   # (1350, 24)

print(datasets.shape)    # (1350, 24) -> (1350, 23)
print(test_sets.shape)   # (1350, 24) -> (1350, 22)

######## xy 분리
x = datasets.drop(['target'], axis=1)
y = datasets['target']

test_sets = test_sets.values    # 넘파이로 변경

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

######################################### y값 로그변환 ###########################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# # scaler = PowerTransformer(method='box-cox')  # error
scaler = PowerTransformer(method='yeo-johnson')  # 디폴트

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_sets = scaler.transform(test_sets)

ak_model = ak.StructuredDataRegressor(
    overwrite=True, max_trials=20, loss='mean_absolute_error'
)

ak_model.fit(x_train, y_train)#, epochs=10, validation_split=0.2)

model = ak_model.export_model()   # 가장 좋은값 뽑아줌
# save model

y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print("loss : ", np.round(results, 6))

print(y_test.shape, y_predict.shape)   # (270,) (270, 1)
y_predict = y_predict.reshape(270, )

nmae = NMAE(np.expm1(y_test), np.expm1(y_predict))
print("NMAE : ", round(nmae, 6))

#loss :  [ 10.05348  101.791885]
#(270,) (270, 1)
#NMAE :  0.999924
"""
####################################################################################################################
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import time 
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor
import autokeras as ak
# loc(인덱스와 컬럼명), iloc(인덱스의 수치) 

####################################### 아웃 라이어 함수 ########################################
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print('1사분위 : ', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr :', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
######################################################

############################################## NMAE 함수 ###################################################
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
########################################################################################################

# object는 최상위 >> string으로 생각해야 함 

path = '../_data/dacon/housing/'
datasets = pd.read_csv(path + 'train.csv', index_col= 0, header=0)
test_sets = pd.read_csv(path + 'test.csv', index_col= 0, header=0)
submit_sets = pd.read_csv(path + 'sample_submission.csv', index_col= 0, header=0)
print(datasets.info())
print(datasets.describe())
print(datasets.isnull().sum())

############################################# 중복값 처리 ###############################################
print('중복값 제거 전', datasets.shape)
datasets = datasets.drop_duplicates()
print('중복값 제거 후', datasets.shape)

########################################### 이상치 확인 및 처리 ###########################################
outliers_loc = outliers(datasets['Garage Yr Blt'])
print('이상치의 위치 :', outliers_loc)
print('이상치 :',datasets.loc[[255], 'Garage Yr Blt'])   # 행과 열  // 2207(이상치)
datasets.drop(datasets[datasets['Garage Yr Blt']==2207].index, inplace = True)
print(datasets['Exter Qual'].value_counts())

'''
1사분위 :  1961.0
q2 : 1978.0
3사분위 : 2002.0
iqr : 41.0
이상치의 위치 : (array([254], dtype=int64),)
'''

print(datasets['Exter Qual'].value_counts())
'''
TA    808
Gd    485
Ex     49
Fa      8
'''
print(datasets['Kitchen Qual'].value_counts())
'''
TA    660
Gd    560
Ex    107
Fa     23
'''
print(datasets['Bsmt Qual'].value_counts())
'''
TA    605
Gd    582
Ex    134
Fa     28
Po      1
'''
print(test_sets['Exter Qual'].value_counts())
print(test_sets['Kitchen Qual'].value_counts())
print(test_sets['Bsmt Qual'].value_counts())
'''
test 파일에도 po가 있기 때문에 처리하는 방식에 대해 고민해봐야 한다. 
'''
# 품질 관련 변수 → 숫자로 매핑
qual_cols = datasets.dtypes[datasets.dtypes == np.object].index
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

datasets = label_encoder(datasets, qual_cols)
test_sets = label_encoder(test_sets, qual_cols)
datasets.head()
print(datasets.shape)  # (1350, 14)
print(test_sets.shape) # (1350, 13)

############################### 분류형 컬럼을 one hot encoding ###########################
datasets = pd.get_dummies(datasets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test_sets = pd.get_dummies(test_sets, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
############################### 분류형 컬럼을 one hot encoding ###########################
# print(datasets.columns)
# print(test_sets.columns)
print(datasets.shape)  # (1350, 23)
print(test_sets.shape) # (1350, 22)

########## xy분리 
x = datasets.drop(['target'], axis=1)
y = datasets['target']

test_sets = test_sets.values   # numpy로 변환
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.8, random_state = 66
)
################ y값 로그변환 ######################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
################ y값 로그변환 ######################

print(x_train.shape, y_train.shape)     # (1080, 22) (1080,)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer(method = 'box-cox')
scaler = PowerTransformer(method = 'yeo-johnson')  # 디폴트

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_sets = scaler.transform(test_sets)

ak_model = ak.StructuredDataRegressor(
    overwrite = True, max_trials = 10, loss= 'mean_absolute_error',
)

start = time.time()
ak_model.fit(x_train, y_train, epochs = 10000, validation_split = 0.2)
end = time.time() - start

model = ak_model.export_model()   # trial의 수만큼 훈련 시킨 것 중에 가장 좋은 것을 꺼낸다.
# 모델 저장하는 법 
y_pred = ak_model.predict(x_test)
results = model.evaluate(x_test,y_test)
print("loss :", np.round(results, 6))
y_pred = y_pred.reshape(270,)

# print(y_test.shape, y_pred.shape)  # (270,) (270, 1)

nmae = NMAE(np.expm1(y_test), np.expm1(y_pred))
print('nmae :', np.round(nmae, 6))

############################################### 제출용 ###############################################
colsample_bytree = 0.8819 
learning_rate = 0.05
max_depth = 4
min_child_weight = 1.7166
n_estimators = 7421
reg_lambda = 8.355
subsample = 0.9054

y_submit = model.predict(test_sets)
y_submit = np.expm1(y_submit)
submit_sets.target = y_submit

path_save_csv = './hackathon/dacon/housing/_save_csv/'
now1 = datetime.now()
now_date = now1.strftime("%m%d_%H%M")  # 연도와 초를 뺴본다

submit_sets.to_csv(path_save_csv + now_date + '_' + str(round(nmae, 4)) + '.csv')

model.summary()

with open(path_save_csv + now_date + '_' + str(round(nmae, 4)) + 'submit.txt', 'a') as file:
    file.write("\n=================================================================")  # \n : 한줄 띄어씀
    file.write('저장시간 : ' + now_date + '\n')
    file.write('scaler : ' + str(scaler) + '\n')
    file.write('colsample_bytree : ' + str('colsample_bytree') + '\n')
    file.write('learning_rate : ' + str('learning_rate') + '\n')
    file.write('max_depth : ' + str('max_depth') + '\n')
    file.write('min_child_weight : ' + str('min_child_weight') + '\n')
    file.write('n_estimators : ' + str('n_estimators') + '\n')
    file.write('reg_lambda : ' + str('reg_lambda') + '\n')
    file.write('subsample : ' + str('subsample') + '\n')

    file.write('걸린시간 : ' + str(round(end, 4)) + '\n')
    file.write('evaluate : ' + str(np.round(results, 4)) + '\n')
    file.write('NMAE : '+ str(round(nmae, 4)) + '\n')
    #file.close()
    
    
    
    