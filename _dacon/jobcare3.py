import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import seaborn as sns

DATA_PATH = 'D:\_data\dacon\Jobcare_data'
# id column은 필요없으니 제거
train_data = pd.read_csv(f'{DATA_PATH}train.csv')
test_data = pd.read_csv(f'{DATA_PATH}test.csv')
sample_submission = pd.read_csv(f'{DATA_PATH}sample_submission.csv')

print(train_data.head)
'''
<bound method NDFrame.head of             id  d_l_match_yn  d_m_match_yn  d_s_match_yn  h_l_match_yn  h_m_match_yn  ...  contents_attribute_e  contents_attribute_h  person_rn  contents_rn     contents_open_dt  target
0            0          True          True          True         False         False  ...                     4                   139     618822       354805  2020-01-17 12:09:36       1
1            1         False         False         False          True          True  ...                     4                   133     571659       346213  2020-06-18 17:48:52       0
2            2         False         False         False          True         False  ...                     4                    53     399816       206408  2020-07-08 20:00:10       0
3            3         False         False         False          True         False  ...                     3                    74     827967       572323  2020-01-13 18:09:34       0
4            4          True          True          True         False         False  ...                     4                    74     831614       573899  2020-03-09 20:39:22       0
501946  501946         False         False         False          True         False  ...                     5                    65     503156       285850  2020-03-13 12:55:52       1
501947  501947          True          True         False          True         False  ...                     4                   142     676255       456996  2020-01-20 11:51:51       1
501948  501948          True          True          True          True         False  ...                     7                    65     484528       293258  2020-08-05 17:27:24       1
501949  501949          True         False         False          True         False  ...                     4                   259     456330       273797  2020-06-15 09:23:21       1
501950  501950          True          True          True          True         False  ...                     5                   289     235596       176650  2020-05-25 14:34:48       1
[501951 rows x 35 columns]>
'''
code_d = pd.read_csv(f'{DATA_PATH}속성_D_코드.csv').iloc[:,:-1]
code_h = pd.read_csv(f'{DATA_PATH}속성_H_코드.csv')
code_l = pd.read_csv(f'{DATA_PATH}속성_L_코드.csv')

code_d.columns= ["attribute_d_d","attribute_d_s","attribute_d_m","attribute_d_l"]
code_h.columns= ["attribute","attribute_h","attribute_h_p"]
code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l"]


def merge_codes(df:pd.DataFrame,df_code:pd.DataFrame,col:str)->pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df,df_code,how="left",on=col)

import os
import sys
import platform
import random
import math
from typing import List ,Dict, Tuple

def preprocess_data(
                    df:pd.DataFrame,is_train:bool = True, cols_merge:List[Tuple[str,pd.DataFrame]] = []  , cols_equi:List[Tuple[str,str]]= [] ,
                    cols_drop:List[str] = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt"]
                    )->Tuple[pd.DataFrame,np.ndarray]:
    df = df.copy()

    y_data = None
    if is_train:
        y_data = df["target"].to_numpy()
        df = df.drop(columns="target")

    for col, df_code in cols_merge:
        df = merge_codes(df,df_code,col)

    cols = df.select_dtypes(bool).columns.tolist()
    df[cols] = df[cols].astype(int)

    for col1, col2 in cols_equi:
        df[f"{col1}_{col2}"] = (df[col1] == df[col2] ).astype(int)

    df = df.drop(columns=cols_drop)
    return (df , y_data)
  
# 소분류 중분류 대분류 속성코드 merge 컬럼명 및 데이터 프레임 리스트
cols_merge = [
              ("person_prefer_d_1" , code_d),
              ("person_prefer_d_2" , code_d),
              ("person_prefer_d_3" , code_d),
              ("contents_attribute_d" , code_d),
              ("person_prefer_h_1" , code_h),
              ("person_prefer_h_2" , code_h),
              ("person_prefer_h_3" , code_h),
              ("contents_attribute_h" , code_h),
              ("contents_attribute_l" , code_l),
]

# 회원 속성과 콘텐츠 속성의 동일한 코드 여부에 대한 컬럼명 리스트
cols_equi = [

    ("contents_attribute_c","person_prefer_c"),
    ("contents_attribute_e","person_prefer_e"),

    ("person_prefer_d_2_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_2_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_2_attribute_d_l" , "contents_attribute_d_attribute_d_l"),
    ("person_prefer_d_3_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_3_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_3_attribute_d_l" , "contents_attribute_d_attribute_d_l"),

    ("person_prefer_h_1_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_2_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_3_attribute_h_p" , "contents_attribute_h_attribute_h_p"),

]

# 학습에 필요없는 컬럼 리스트
cols_drop = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt", "contents_rn", ]  


x_train, y_train = preprocess_data(train_data, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
x_test, _ = preprocess_data(test_data,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
x_train.shape , y_train.shape , x_test.shape

cat_features = x_train.columns[x_train.nunique() > 2].tolist()



# for col in train_data.columns:
#     n_nan = train_data[col].isnull().sum()
#     if n_nan>0:
#       msg = '{:^20}에서 결측치 개수 : {}개'.format(col,n_nan)
#       print(msg)

# #         Sex         에서 결측치 개수 : 3개
# #  Delta 15 N (o/oo)  에서 결측치 개수 : 3개
# #  Delta 13 C (o/oo)  에서 결측치 개수 : 3개


# for col in test_data.columns:
#     n_nan = test_data[col].isnull().sum()
#     if n_nan>0:
#       msg = '{:^20}에서 결측치 개수 : {}개'.format(col,n_nan)
#       print(msg)

# #         Sex         에서 결측치 개수 : 6개
# #  Delta 15 N (o/oo)  에서 결측치 개수 : 9개
# #  Delta 13 C (o/oo)  에서 결측치 개수 : 8개


# 다양한 알고리즘 비교를 통해 성별 예측을 잘하는 최선의 알고리즘 찾기.
models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class = 'ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
models.append(('RFC',RandomForestClassifier()))
models.append(('XGBC',XGBClassifier(iterations=10000,verbose=False)))
models.append(('LGBMC',LGBMClassifier()))
models.append(('AdaC',AdaBoostClassifier()))
models.append(('Cat',CatBoostClassifier(iterations=10000,verbose=False)))
results =[]
names = []


is_holdout = False
n_splits = 5
iterations = 3000
patience = 50

cv = KFold(n_splits=n_splits, shuffle=True, random_state=66)
scores = []
models = []


models = []
for tri, vai in cv.split(x_train):
    print("="*50)
    preds = []

    model = CatBoostClassifier(iterations=iterations,random_state=66,task_type="GPU",eval_metric="F1",cat_features=cat_features,one_hot_max_size=4)
    model.fit(x_train.iloc[tri], y_train[tri], 
            eval_set=[(x_train.iloc[vai], y_train[vai])], 
            early_stopping_rounds=patience ,
            verbose = 100
        )
    
    models.append(model)
    scores.append(model.get_best_score()["validation"]["F1"])
    if is_holdout:
        break    

# for name, model in models:
#   kfold = KFold(n_splits=10,random_state=7,shuffle = True)
#   cv_results = cross_val_score(model,x_train.iloc[tri], y_train[tri]
#                                ,cv= kfold,scoring='accuracy')
#   results.append(cv_results)
#   names.append(name)
#   msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
#   print(msg)

num_folds= 10
seed = 7
scoring = 'neg_root_mean_squared_error'
X_all = x_train.iloc[tri]
y_all = y_train[tri]

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all
                                                      ,test_size=0.3,random_state=66)


models = []
# models.append(('LR',LinearRegression()))
# models.append(('LASSO',Lasso()))
# models.append(('KNN',KNeighborsRegressor()))
# models.append(('CART',DecisionTreeRegressor()))
# models.append(('EN',ElasticNet()))
# models.append(('SVM',SVR()))
# models.append(('RFR',RandomForestRegressor()))
models.append(('XGBR',XGBRegressor()))
models.append(('LGBMR',LGBMRegressor()))
# models.append(('AdaR',AdaBoostRegressor()))
models.append(('Cat',CatBoostRegressor(verbose=False)))
# models.append(('Xtree',ExtraTreesRegressor()))

results =[]
names = []
for name, model in models:
  kfold = KFold(n_splits=10,random_state=66,shuffle = True)
  cv_results = cross_val_score(model,X_train,y_train
                               ,cv= kfold,scoring=scoring)#, n_jobs=-1)
  results.append(cv_results)
  names.append(name)
  msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
  print(msg)




#standardization/CatBoostClassifier

pipelines = []
# pipelines.append(('ScaledLR',Pipeline([('Scaler',preprocessing.StandardScaler()),('LR',LinearRegression())])))
# pipelines.append(('ScaledLASSO',Pipeline([('Scaler',preprocessing.StandardScaler()),('LASSO',Lasso())])))
# pipelines.append(('ScaledKNN',Pipeline([('Scaler',preprocessing.StandardScaler()),('KNN',KNeighborsRegressor())])))
# pipelines.append(('ScaledCART',Pipeline([('Scaler',preprocessing.StandardScaler()),('CART',DecisionTreeRegressor())])))
# pipelines.append(('ScaledEN',Pipeline([('Scaler',preprocessing.StandardScaler()),('EN',ElasticNet())])))
# pipelines.append(('ScaledSVM',Pipeline([('Scaler',preprocessing.StandardScaler()),('SVM',SVR())])))
# pipelines.append(('ScaledRFR',Pipeline([('Scaler',preprocessing.StandardScaler()),('RFR',RandomForestRegressor())])))
pipelines.append(('ScaledXGBR',Pipeline([('Scaler',preprocessing.StandardScaler()),('XGBR',XGBRegressor())])))
pipelines.append(('ScaledLGBMR',Pipeline([('Scaler',preprocessing.StandardScaler()),('LGBMR',LGBMRegressor())])))
# pipelines.append(('ScaledAdaR',Pipeline([('Scaler',preprocessing.StandardScaler()),('AdaR',AdaBoostRegressor())])))
pipelines.append(('ScaledCat',Pipeline([('Scaler',preprocessing.StandardScaler()),('Cat',CatBoostRegressor(verbose=False))])))
# pipelines.append(('ScaledXtree',Pipeline([('Scaler',preprocessing.StandardScaler()),('Xtree',ExtraTreesRegressor())])))

results_scaled =[]
names_scaled = []
for name, model in pipelines:
  kfold = KFold(n_splits=10,random_state=66,shuffle = True)
  cv_results = cross_val_score(model,X_train,y_train
                               ,cv= kfold,scoring=scoring, n_jobs=-1)
  results_scaled.append(cv_results)
  names_scaled.append(name)
  msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
  print(msg)
   
  
scaler = preprocessing.StandardScaler().fit(X_all)
scaled_X = scaler.transform(X_all)

# params = { 'n_estimators' : [10, 50,100],
#            'max_depth' : [6, 12,18,24],
#            'min_samples_leaf' : [1, 6, 12, 18],
#            'min_samples_split' : [2, 8, 16, 20]
#             }
# model = RandomForestRegressor()
# kfold = KFold(n_splits= num_folds,random_state = 66 ,shuffle = True)
# grid = GridSearchCV(estimator= model, param_grid = params,scoring= 'neg_root_mean_squared_error',cv=kfold )
# grid_result = grid.fit(scaled_X,y_all)

# print("Best : %f using %s "%(grid_result.best_score_,grid_result.best_params_))  
  
  
  
  
# params = { 'n_estimators' : [10, 50,100],
#            'max_depth' : [6,12,18,24],
#            'min_samples_leaf' : [1, 6, 12, 18],
#            'min_samples_split' : [2,4,8, 16]
#             }
# model =ExtraTreesRegressor()
# kfold = KFold(n_splits= num_folds,random_state = 66 ,shuffle = True)
# grid = GridSearchCV(estimator= model, param_grid = params, scoring= 'neg_root_mean_squared_error',cv=kfold )
# grid_result = grid.fit(X_all,y_all)  
  
  

# print("Best : %f using %s "%(grid_result.best_score_,grid_result.best_params_))

  
  
from sklearn.metrics import mean_squared_error
import math   
  
  
errors = []
pred_valid=[]
pred_test = []  
# test = y_train[tri]
  
scaler = preprocessing.StandardScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_valid = scaler.transform(X_valid)
scaled_X_test = scaler.transform(y_train[tri]) 
  

# lasso = Lasso()
# lasso.fit(X_train,y_train)
# lasso_valid = lasso.predict(X_valid)
# rmse = math.sqrt(mean_squared_error(y_valid, lasso_valid))
# errors.append(('Lasso',rmse))
# pred_valid.append(('Lasso',lasso_valid))
# lasso_test = lasso.predict(test)
# pred_test.append(('Lasso',lasso_test))  
  
  
# LR =LinearRegression()
# LR.fit(scaled_X_train,y_train)
# lr_valid = LR.predict(scaled_X_valid)
# rmse = math.sqrt(mean_squared_error(y_valid, lr_valid))
# errors.append(('LR',rmse))
# pred_valid.append(('LR',lr_valid))
# lr_test = LR.predict(scaled_X_test)
# pred_test.append(('LR',lr_test))  
  
# RF =RandomForestRegressor(max_depth= 24, min_samples_leaf= 12, min_samples_split= 16, n_estimators= 40)
# RF.fit(scaled_X_train,y_train)
# rf_valid = RF.predict(scaled_X_valid)
# rmse = math.sqrt(mean_squared_error(y_valid, rf_valid))
# errors.append(('RF',rmse))
# pred_valid.append(('RF',rf_valid))
# rf_test = RF.predict(scaled_X_test)
# pred_test.append(('RF',rf_test))  
  
# ET =ExtraTreesRegressor(max_depth=24, min_samples_leaf= 12, min_samples_split= 16, n_estimators= 40)
# ET.fit(X_train,y_train)
# et_valid = ET.predict(X_valid)
# rmse = math.sqrt(mean_squared_error(y_valid, et_valid))
# errors.append(('ET',rmse))
# pred_valid.append(('ET',et_valid))
# et_test = ET.predict(test)
# pred_test.append(('ET',et_test))  
  
CAT = CatBoostRegressor(iterations=10000,random_state=66
           ,eval_metric="RMSE")
CAT.fit(X_train,y_train, eval_set=[(X_valid,y_valid)],early_stopping_rounds=30
        ,verbose=1000 )
cat_valid = CAT.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, cat_valid))
errors.append(('CAT',rmse))
pred_valid.append(('CAT',cat_valid))
cat_test = CAT.predict(y_train[tri])
pred_test.append(('CAT',cat_test))
  
  
for name, error in errors:
      print("{} : {}".format(name,error)) 
  
  
val= np.zeros(X_valid.shape[0])
for name, pred in pred_valid:
  val+= (0.2* pred)
math.sqrt(mean_squared_error(y_valid, val))  
  
  
val= np.zeros(X_valid.shape[0])
for name, pred in pred_valid:
  if name == 'Lasso' or name=='LR' or name == 'ET' or name=='CAT':
    val+= (0.25* pred)
math.sqrt(mean_squared_error(y_valid, val)) 
  
  

test_val= np.zeros(test.shape[0])
for name, pred in pred_test:
  if name == 'Lasso' or name=='LR' or name == 'ET' or name=='CAT':
    test_val+= (0.25* pred)

#model.save_weights("./_save/keras999_1_save_weights.h5")
#model = load_model('./_ModelCheckPoint/ss_ki_1222_Trafevol5.hdf5')
sample_submission = pd.read_csv(DATA_PATH+'sample_submission.csv')
sample_submission['Body Mass (g)'] = test_val
sample_submission.to_csv(DATA_PATH+"jobcare_0117_2.csv", index=False)

  
# sample_submission['Body Mass (g)'] = test_val
# sample_submission.to_csv(DATA_PATH+"jobcare_0117_1.csv", index=False)
