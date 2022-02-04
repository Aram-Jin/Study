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

# {'target': 0.08905597988490348, 
#  'params': {'colsample_bytree': 0.8819250794365909, 
#             'learning_rate': 0.11901206948192891, 
#             'max_depth': 5.002515171899393, 
#             'min_child_weight': 1.7166758886339086, 
#             'n_estimators': 7421.505876139617, 
#             'reg_lambda': 8.355049478293875, 
#             'subsample': 0.9054715831126499}}   

colsample_bytree = 0.8819 
learning_rate = 0.119 
max_depth = 5
min_child_weight = 1.7166
n_estimators = 7421
reg_lambda = 8.355
subsample = 0.9054

############################### 여기부터 SelectFromModel ###############################
model = XGBRegressor(n_jobs=-1,
                     colsample_bytree=colsample_bytree,
                     learning_rate = learning_rate,
                     max_depth = max_depth,
                     min_child_weight = min_child_weight,
                     n_estimators = n_estimators,
                     reg_lambda = reg_lambda,
                     subsample = subsample
                     )

model.fit(x_train, y_train,
          early_stopping_rounds=100,
          eval_set=[(x_test, y_test)],
          eval_metric='mae')

############################### SelectFromModel 적용!! ######################################
print(datasets.columns)
# Index(['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
#        'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built',
#        'Year Remod/Add', 'Garage Yr Blt', 'target', 'Exter Qual_2',
#        'Exter Qual_3', 'Exter Qual_4', 'Exter Qual_5', 'Kitchen Qual_2',
#        'Kitchen Qual_3', 'Kitchen Qual_4', 'Kitchen Qual_5', 'Bsmt Qual_2',
#        'Bsmt Qual_3', 'Bsmt Qual_4', 'Bsmt Qual_5'],
#       dtype='object')
# thresholds = np.sort(model.feature_importances_)    # 0.005보다 작은 것 4개 삭제
thresholds = model.feature_importances_
print(thresholds)
# [0.36020315 0.04666024 0.00492802 0.00666874 0.01227359 0.00852228
#  0.04635533 0.00711413 0.00853281 0.0061158  0.01088824 0.00399592
#  0.034023   0.00309655 0.00460793 0.02265866 0.0038926  0.00743985
#  0.02046973 0.00772257 0.03057562 0.34325537]  -> 0.005보다 작은 것 4개는 11,13,14,16번째(순서 셀때는 0부터 시작)

x_train = np.delete(x_train, [11,13,14,16], axis=1)
x_test = np.delete(x_test, [11,13,14,16], axis=1)
test_sets = np.delete(test_sets, [11,13,14,16], axis=1)

print(x_train.shape, x_test.shape, test_sets.shape)   # (1078, 18) (270, 18) (1350, 18)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1,
                                    colsample_bytree=colsample_bytree,
                     learning_rate = learning_rate,
                     max_depth = max_depth,
                     min_child_weight = min_child_weight,
                     n_estimators = n_estimators,
                     reg_lambda = reg_lambda,
                     subsample = subsample)
    
    selection_model.fit(select_x_train, y_train,
                        early_stopping_rounds=100,
          eval_set=[(select_x_test, y_test)],
          eval_metric='mae',
          verbose=0)
    
    y_predict = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))
    
# (1078, 22) (270, 22)
# Thresh=0.003, n=22, R2: 90.17%
# (1078, 21) (270, 21)
# Thresh=0.004, n=21, R2: 89.69%
# (1078, 20) (270, 20)
# Thresh=0.004, n=20, R2: 89.40%
# (1078, 19) (270, 19)
# Thresh=0.005, n=19, R2: 89.80%
# (1078, 18) (270, 18)
# Thresh=0.005, n=18, R2: 89.95%   -> 이것을 기준으로 낮은 것들 삭제
# (1078, 17) (270, 17)
# Thresh=0.006, n=17, R2: 89.38%
# (1078, 16) (270, 16)
# Thresh=0.007, n=16, R2: 89.15%
# (1078, 15) (270, 15)
# Thresh=0.007, n=15, R2: 87.83%
# (1078, 14) (270, 14)
# Thresh=0.007, n=14, R2: 87.97%
# (1078, 13) (270, 13)
# Thresh=0.008, n=13, R2: 87.52%
# (1078, 12) (270, 12)
# Thresh=0.009, n=12, R2: 87.70%
# (1078, 11) (270, 11)
# Thresh=0.009, n=11, R2: 88.09%
# (1078, 10) (270, 10)
# Thresh=0.011, n=10, R2: 87.17%
# (1078, 9) (270, 9)
# Thresh=0.012, n=9, R2: 87.29%
# (1078, 8) (270, 8)
# Thresh=0.020, n=8, R2: 83.81%
# (1078, 7) (270, 7)
# Thresh=0.023, n=7, R2: 83.48%
# (1078, 6) (270, 6)
# Thresh=0.031, n=6, R2: 84.45%
# (1078, 5) (270, 5)
# Thresh=0.034, n=5, R2: 83.58%
# (1078, 4) (270, 4)
# Thresh=0.046, n=4, R2: 83.35%
# (1078, 3) (270, 3)
# Thresh=0.047, n=3, R2: 82.97%
# (1078, 2) (270, 2)
# Thresh=0.343, n=2, R2: 70.10%
# (1078, 1) (270, 1)
# Thresh=0.360, n=1, R2: 70.44%    
