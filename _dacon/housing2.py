import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

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

#1. 데이터
path = "./housing/"    

train = pd.read_csv(path + 'train.csv')
# print(train.shape)  # (1350, 15)
test = pd.read_csv(path + 'test.csv')
# # print(test_file.shape)  # (1350, 14)
submission = pd.read_csv(path + 'sample_submission.csv')#, dtype=float
# print(submit_file.shape)  # (1350, 2)

train = train.iloc[:, 1:]
test = test.iloc[:, 1:]

print(train.info())
print(test.info())

# 중복값 제거
train = train.drop_duplicates()

# 이상치 처리
outliers_loc = outliers(train['Garage Yr Blt'])
print(outliers_loc)
print(train.loc[[255], 'Garage Yr Blt'])   # 2207
train.drop(train[train['Garage Yr Blt']==2207].index, inplace=True)
print(train.shape) 

# train = train[train['Garage Yr Blt'] < 2050]

# cat_cols = train.dtypes[train.dtypes == np.object].index
# print(cat_cols)
# # Index(['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'], dtype='object')

cat_cols = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual']

for c in cat_cols :
    ord_df = train.groupby(c).target.median().reset_index(name = f'ord_{c}')
    train = pd.merge(train, ord_df, how = 'left')
    test = pd.merge(test, ord_df, how = 'left')

train.drop(cat_cols, axis = 1, inplace = True)
test.drop(cat_cols, axis = 1, inplace = True)    

print(f'로그 변환 전 타겟 왜도 = {train.target.skew()} / 로그 변환 후 타겟 왜도 = {np.log1p(train.target).skew()}')

x = train.drop('target', axis = 1)
y = np.log1p(train.target)

target = test[x.columns]

target.fillna(target.mean(), inplace = True)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from ngboost import NGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

def NMAE(true, pred) -> float:
    mae = np.mean(np.abs(true - pred))
    score = mae / np.mean(np.abs(true))
    return score

nmae_score = make_scorer(NMAE, greater_is_better=False)
kf = KFold(n_splits = 10, random_state = 42, shuffle = True)


rf_pred = np.zeros(target.shape[0])
rf_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    rf = RandomForestRegressor(random_state = 42, criterion = 'mae')
    rf.fit(tr_x, tr_y)
    
    val_pred = np.expm1(rf.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    rf_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = rf.predict(target) / 10
    rf_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(rf_val)} & std = {np.std(rf_val)}')

gbr_pred = np.zeros(target.shape[0])
gbr_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    gbr = XGBRegressor(random_state = 42, max_depth = 5, learning_rate = 0.01855, n_estimators = 8000)
    gbr.fit(tr_x, tr_y)
    
    val_pred = np.expm1(gbr.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    gbr_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = gbr.predict(target) / 10
    gbr_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(gbr_val)} & std = {np.std(gbr_val)}')

cb_pred = np.zeros(target.shape[0])
cb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    tr_data = Pool(data = tr_x, label = tr_y)
    val_data = Pool(data = val_x, label = val_y)
    
    cb = CatBoostRegressor(depth = 4, random_state = 42, loss_function = 'MAE', n_estimators = 6000, learning_rate = 0.0215, verbose = 0)
    cb.fit(tr_data, eval_set = val_data, early_stopping_rounds = 750, verbose = 1000)
    
    val_pred = np.expm1(cb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    cb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = cb.predict(target) / 10
    cb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(cb_val)} & std = {np.std(cb_val)}')

ngb_pred = np.zeros(target.shape[0])
ngb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    ngb = NGBRegressor(random_state = 42, n_estimators = 6000, verbose = 0, learning_rate = 0.0215)
    ngb.fit(tr_x, tr_y, val_x, val_y, early_stopping_rounds = 750)
    
    val_pred = np.expm1(ngb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    ngb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = ngb.predict(target) / 10
    ngb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(ngb_val)} & std = {np.std(ngb_val)}')

(ngb_pred + cb_pred + rf_pred + gbr_pred) / 4

submission['target'] = np.expm1((ngb_pred + cb_pred + rf_pred + gbr_pred) / 4)

submission.to_csv('final13.csv', index = False)

# final3 : 10FOLD Mean of NMAE = 0.09514011171959022 & std = 0.009827818836972573
# final4 : 10FOLD Mean of NMAE = 0.09488459045057565 & std = 0.009237313029344942
# final5 : 10FOLD Mean of NMAE = 0.09503054483243782 & std = 0.009392146619015534
# final6 : 10FOLD Mean of NMAE = 0.09507041683386622 & std = 0.009450277486941779
# final7 : 10FOLD Mean of NMAE = 0.0950289114885935 & std = 0.009385066244711497
# final8 : 10FOLD Mean of NMAE = 0.09493117715478769 & std = 0.009466013682820807
# final9 : 10FOLD Mean of NMAE = 0.09468572850389949 & std = 0.00901473785673848  **
# final10 : 10FOLD Mean of NMAE = 0.09496452258027831 & std = 0.009360689609836432
# final11 : 10FOLD Mean of NMAE = 0.09495534834421646 & std = 0.009142361318684754
# final12 : 10FOLD Mean of NMAE = 0.09485109375225506 & std = 0.009167514515949037
# final13 : 10FOLD Mean of NMAE = 0.09481274389058796 & std = 0.00902713942718833