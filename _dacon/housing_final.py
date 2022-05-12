from tabnanny import verbose
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor

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

path = './housing/'
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

########################## Bayesian Optimization 쓰잣!!!!! #####################################
parms = {'max_depth': (2,12),
         'learning_rate': (0.025, 0.255),
         'n_estimators': (5000, 10000),
        #  'gamma': (0, 100),
         'min_child_weight': (0, 1000),
         'subsample': (0.5, 1),
         'colsample_bytree': (0, 1),
         'reg_lambda': (0.001, 10),
        #  'reg_alpha': (0.01, 50)
         }

def xg_def(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, reg_lambda):
      xg_model = XGBRegressor(max_depth = int(max_depth),
                              learning_rate = learning_rate,
                              n_estimators = int(n_estimators),
                              min_child_weight = min_child_weight,
                              subsample = subsample,
                              colsample_bytree = colsample_bytree,
                              reg_lambda = reg_lambda
                              )
      xg_model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                   eval_metric='mae',
                   verbose=1,
                   early_stopping_rounds=80)
      y_predict = xg_model.predict(x_test)
      
      nmae = NMAE(y_test, y_predict)
      return nmae
    
bo = BayesianOptimization(f=xg_def, pbounds=parms, random_state=66, verbose=2)

bo.maximize(init_points=300, n_iter=1000)
# n_iter : 수행하려는 베이지안 최적화 단계. 더 많은 단계를 거치면 더 좋은 최대치 얻음
# init_points : 수행할 무작위 탐색 단계

print("============================= bo.res =================================")
print(bo.res)
print("============================= 파라미터 튜닝 결과 =================================")
print(bo.max)    

target_list=[]
for result in bo.res:
      target = result['target']
      target_list.append(target)
      
min_dict = bo.res[np.argmin(np.array(target_list))]
print(min_dict)

# {'target': 0.08905597988490348, 
#  'params': {'colsample_bytree': 0.8819250794365909, 
#             'learning_rate': 0.11901206948192891, 
#             'max_depth': 5.002515171899393, 
#             'min_child_weight': 1.7166758886339086, 
#             'n_estimators': 7421.505876139617, 
#             'reg_lambda': 8.355049478293875, 
#             'subsample': 0.9054715831126499}}   


