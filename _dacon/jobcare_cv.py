path = "../_data/dacon/Jobcare_data/" 
SUBMIT_PATH = "../_data/dacon/Jobcare_data/" 
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

for col1 in code_h.columns:
    n_nan1 = code_h[col1].isnull().sum()
    if n_nan1>0:
      msg1 = '{:^20}에서 결측치 개수: {}개'.format(col1,n_nan1)
      print(msg1)
    else:
        print('결측치가 없습니다.')

for col2 in code_h.columns:
    n_nan2 = code_h[col2].isnull().sum()
    if n_nan2>0:
        msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
        print(msg2)
    else:
      print('결측치가 없습니다.')
      
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
cols_drop = ["id", "person_prefer_f","person_prefer_g" ,"contents_open_dt", "contents_rn", "person_rn",]

x_train, y_train = preprocess_data(train_data, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
x_test, _ = preprocess_data(test_data,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
x_train.shape , y_train.shape , x_test.shape

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

cat_features = x_train.columns[x_train.nunique() > 2].tolist()

is_holdout = False
n_splits = 3
iterations = 3000
patience = 600

cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

scores = []
models = []
for tri, vai in cv.split(x_train):
    print("="*50)
    preds = []

    model = CatBoostClassifier(iterations=iterations, random_state=SEED, learning_rate=0.15, depth= 6, task_type="GPU",eval_metric="F1",cat_features=cat_features,one_hot_max_size=4)
    
    # 'iterations': 300,
    # 'learning_rate': 0.01,
    # 'depth': 16,
    # 'eval_metric': 'Logloss',
    
    # iterations=None, 
    # learning_rate=None, 
    # depth=None, l2_leaf_reg=None, 
    # model_size_reg=None, rsm=None, 
    # loss_function=None, border_count=None, 
    # feature_border_type=None, 
    # per_float_feature_quantization=None, 
    # input_borders=None, output_borders=None, 
    # fold_permutation_block=None, od_pval=None, 
    # od_wait=None, od_type=None, nan_mode=None, 
    # counter_calc_method=None, leaf_estimation_iterations=None, 
    # leaf_estimation_method=None, thread_count=None, 
    # random_seed=None, use_best_model=None, 
    # best_model_min_trees=None, verbose=None, silent=None, 
    # logging_level=None, metric_period=None, 
    # ctr_leaf_count_limit=None, store_all_simple_ctr=None, 
    # max_ctr_complexity=None, has_time=None, allow_const_label=None, 
    # target_border=None, classes_count=None, class_weights=None, 
    # auto_class_weights=None, class_names=None, one_hot_max_size=None, 
    # random_strength=None, name=None, ignored_features=None, 
    # train_dir=None, custom_loss=None, custom_metric=None, 
    # eval_metric=None, bagging_temperature=None, save_snapshot=None, 
    # snapshot_file=None, snapshot_interval=None, fold_len_multiplier=None, 
    # used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, 
    # allow_writing_files=None, final_ctr_computation_mode=None, 
    # approx_on_full_history=None, boosting_type=None, simple_ctr=None, 
    # combinations_ctr=None, per_feature_ctr=None, ctr_description=None, 
    # ctr_target_border_count=None, task_type=None, device_config=None, 
    # devices=None, bootstrap_type=None, subsample=None, mvs_reg=None, 
    # sampling_unit=None, sampling_frequency=None, dev_score_calc_obj_block_size=None, 
    # dev_efb_max_buckets=None, sparse_features_conflict_fraction=None, max_depth=None, 
    # n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, 
    # random_state=None, reg_lambda=None, objective=None, eta=None, max_bin=None, 
    # scale_pos_weight=None, gpu_cat_features_storage=None, data_partition=None, 
    # metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None, 
    # min_data_in_leaf=None, min_child_samples=None, max_leaves=None, num_leaves=None, 
    # score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None, 
    # monotone_constraints=None, feature_weights=None, penalties_coefficient=None, 
    # first_feature_use_penalties=None, per_object_feature_penalties=None, model_shrink_rate=None, 
    # model_shrink_mode=None, langevin=None, diffusion_temperature=None, posterior_sampling=None, 
    # boost_from_average=None, text_features=None, tokenizers=None, dictionaries=None, 
    # feature_calcers=None, text_processing=None, embedding_features=None, callback=None
    
    
    model.fit(x_train.iloc[tri], y_train[tri], 
            eval_set=[(x_train.iloc[vai], y_train[vai])], 
            early_stopping_rounds=300 ,
            verbose = 100
        )
    
    models.append(model)
    scores.append(model.get_best_score()["validation"]["F1"])
    if is_holdout:
        break    
    
print(scores)
print(np.mean(scores))

threshold = 0.325

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

sample_submission.to_csv(SUBMIT_PATH + "prediction5.csv", index=False)

# bestTest = 0.6790974705
# bestIteration = 784
# Shrink model to first 785 iterations.
# [0.6815594925220745, 0.6826774709618788, 0.6782435784820023, 0.6767866019194738, 0.6790974705274239]
# 0.6796729228825706
# [0.7101563806021186, 0.7103759260813503, 0.7107932786244626, 0.71202682540415, 0.707087528455533]
# 0.7100879878335229


# bestTest = 0.6789540877
# bestIteration = 984
# Shrink model to first 985 iterations.
# [0.6782602070456472, 0.6788961131683913, 0.6816755823821647, 0.6815771182648273, 0.6789540876637192]
# 0.6798726217049499
# [0.6802147187089664, 0.6806485748421232, 0.6843634478713393, 0.6797882042277485, 0.6796289495518287]
# 0.6809287790404013


# prediction4

# bestTest = 0.6797164341
# bestIteration = 631
# Shrink model to first 632 iterations.
# [0.6784979570167704, 0.6341658590367124, 0.6802567288558377, 0.6367686632331189, 0.6797164341027272]
# 0.6618811284490332
# [0.7093967837942385, 0.6657146462801464, 0.7095819113957582, 0.6663479986987725, 0.7068007244114157]
# 0.6915684129160663


# prediction4
# bestTest = 0.6807594961
# bestIteration = 472
# Shrink model to first 473 iterations.
# [0.6802998145280024, 0.6800211057927864, 0.6817024026763989, 0.6818225493398927, 0.6807594960707548]
# 0.680921073681567
# [0.7082815407447586, 0.7035924349388338, 0.710266742850269, 0.7091394766231022, 0.7064766101593296]
# 0.7075513610632587