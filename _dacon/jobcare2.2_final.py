path = "../Jobcare_data/" 
SUBMIT_PATH = "../Jobcare_data/" 
SEED = 66

import os
import sys
import platform
import random
import math
from typing import List ,Dict, Tuple
import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 
from catboost import Pool,CatBoostClassifier
from imblearn.over_sampling import SMOTE

print(f"- os: {platform.platform()}")
print(f"- python: {sys.version}")
print(f"- pandas: {pd.__version__}")
print(f"- numpy: {np.__version__}")
print(f"- sklearn: {sklearn.__version__}")

train_data = pd.read_csv(path +'train.csv')
test_data = pd.read_csv(path +'test.csv')

code_d = pd.read_csv(path+'속성_D_코드.csv').iloc[:,:-1]
code_h = pd.read_csv(path+'속성_H_코드.csv')
code_l = pd.read_csv(path+'속성_L_코드.csv')

print(train_data.shape , test_data.shape)

code_d.columns= ["attribute_d_d","attribute_d_s","attribute_d_m","attribute_d_l"]
code_h.columns= ["attribute_h_","attribute_h","attribute_h_p"]
code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l"]

def merge_codes(df:pd.DataFrame,df_code:pd.DataFrame,col:str)->pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df,df_code,how="left",on=col)

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
cols_drop = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt", "contents_rn", "person_rn",]

x_train, y_train = preprocess_data(train_data, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
x_test, _ = preprocess_data(test_data,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
x_train.shape , y_train.shape , x_test.shape

smote = SMOTE(random_state=66, k_neighbors=4)
x_train, y_train = smote.fit_resample(x_train, y_train)

cat_features = x_train.columns[x_train.nunique() > 2].tolist()

is_holdout = False
n_splits = 5
iterations = 3000
patience = 150

cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

scores = []
models = []

for tri, vai in cv.split(x_train):
    print("="*50)
    preds = []

    model = CatBoostClassifier(iterations=iterations, random_state=SEED, learning_rate=0.2, depth= 4, task_type="GPU",eval_metric="F1",cat_features=cat_features,one_hot_max_size=4)
    model.fit(x_train.iloc[tri], y_train[tri], 
            eval_set=[(x_train.iloc[vai], y_train[vai])], 
            early_stopping_rounds=60 ,
            verbose = 100
        )
    
    models.append(model)
    scores.append(model.get_best_score()["validation"]["F1"])
    if is_holdout:
        break    
    
print(scores)
print(np.mean(scores))

threshold = 0.356

pred_list = []
scores = []
for i,(tri, vai) in enumerate( cv.split(x_train) ):
    pred = models[i].predict_proba(x_train.iloc[vai])[:, 1]
    pred = np.where(pred >= threshold , 1, 0)
    score = f1_score(y_train[vai],pred)
    scores.append(score)
    pred = models[i].predict_proba(x_test)[:, 1]
    pred_list.append(pred)
print(scores)
print(np.mean(scores))

pred = np.mean( pred_list , axis = 0 )
pred = np.where(pred >= threshold , 1, 0)

sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission['target'] = pred
sample_submission

sample_submission.to_csv(SUBMIT_PATH + "prediction2.csv", index=False)
'''
bestTest = 0.6794575174
bestIteration = 808
Shrink model to first 809 iterations.
[0.6488679941319286, 0.6805371383747642, 0.677811260299054, 0.6770470155705233, 0.6794575174463442]
0.6727441851645228
[0.6676968583574853, 0.7097391455613627, 0.7071624024618982, 0.7099820325026387, 0.7067344934156411]
0.7002629864598051

bestTest = 0.6794639078
bestIteration = 807
Shrink model to first 808 iterations.
[0.6488679941319286, 0.6805371383747642, 0.6764870798209517, 0.6743874192368645, 0.6794639078297672]
0.6719487078788553
[0.6676968583574853, 0.7097391455613627, 0.7061146845931873, 0.7069685888271952, 0.7067429009486697]
0.69945243565758

bestTest = 0.6791772608
[0.6815594925220745, 0.6826742887429252, 0.677377819014978, 0.6769481133860651, 0.6791772608250162]
0.6795473948982117
[0.6883710650993826, 0.6884415152857776, 0.6843955435609993, 0.6878623475641281, 0.6861358362483981]
0.6870412615517372



bestTest = 0.6806456193
bestIteration = 868
Shrink model to first 869 iterations.
[0.6780941918498384, 0.6793202425532723, 0.680622009569378, 0.6825925537093064, 0.6806456193496474]
0.6802549234062886
[0.706928395890892, 0.7082479409148912, 0.7083531786532398, 0.7107789665869828, 0.7094055286757681]
0.7087428021443547
'''