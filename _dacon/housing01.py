import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBRegressor
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.decomposition import PCA

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

#1. 데이터
path = "../_data/dacon/housing/"    

train = pd.read_csv(path + 'train.csv')
# print(train.shape)  # (1350, 15)
test = pd.read_csv(path + 'test.csv')
# # print(test_file.shape)  # (1350, 14)
submit_file = pd.read_csv(path + 'sample_submission.csv')#, dtype=float
# print(submit_file.shape)  # (1350, 2)

print(train.head(5))
print(train['target'].describe())
print(train.head())
print(train.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
print(train.info())
print(train.columns)
# Index(['id', 'Overall Qual', 'Gr Liv Area', 'Exter Qual', 'Garage Cars',
#        'Garage Area', 'Kitchen Qual', 'Total Bsmt SF', '1st Flr SF',
#        'Bsmt Qual', 'Full Bath', 'Year Built', 'Year Remod/Add',
#        'Garage Yr Blt', 'target'],
#       dtype='object')
# print(type(train))   # <class 'pandas.core.frame.DataFrame'>

train = train.drop_duplicates()

outliers_loc = outliers(train['Garage Yr Blt'])
print('이상치의 위치 :', outliers_loc)
print('이상치 :',train.loc[[255], 'Garage Yr Blt'])   # 행과 열  // 2207(이상치)
train.drop(train[train['Garage Yr Blt']==2207].index, inplace = True)
print(train['Exter Qual'].value_counts())

qual_cols = train.dtypes[train.dtypes == np.object].index
print(qual_cols)
# Index(['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'], dtype='object')

def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2}
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

train = label_encoder(train, qual_cols)
test = label_encoder(test, qual_cols)

print(train.shape)   # (1350, 14)
print(test.shape)   # (1350, 13)

########################################## 분류형 컬럼을 one hot encoding #########################################
train = pd.get_dummies(train, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test = pd.get_dummies(test, columns=['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])

########################################## 분류형 컬럼을 one hot encoding #########################################
# print(datasets.columns)  
# print(test_sets.columns)   # (1350, 24)

print(train.shape)    # (1350, 24) -> (1350, 23)
print(test.shape)   # (1350, 24) -> (1350, 22)

# train.replace(['Ex'],5,inplace=True)
# train.replace(['Gd'],4,inplace=True)
# train.replace(['TA'],3,inplace=True)
# train.replace(['Fa'],2,inplace=True)
# train.replace(['Po'],1,inplace=True)

# print(train.head(3))

x = train.drop(['id','target'], axis=1)
y = train['target']

test = test.values   

# # print(x.shape)  # (1350, 13)
# # print(y.shape)  # (1350,)

# # # # print(np.unique(y, return_counts=True))   

# pca = PCA(n_components=8)
# x = pca.fit_transform(x)
# print(x.shape)   # (1350, 8)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# # # def cut_outlier(df2, columns):
# # #     df=df2.copy()
# # #     for column in columns:
# # #         q1=df[column].quantile(.25)
# # #         q3=df[column].quantile(.75)
# # #         iqr=q3-q1
# # #         low=q1-1.5*iqr
# # #         high=q3+1.5*iqr
# # #         df.loc[df[column]<low, column]=low
# # #         df.loc[df[column]>high, column]=high
# # #     return df

# # # train_data_2=cut_outlier(train_data, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])
# # # test_data_2=cut_outlier(test_data, ['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area'])

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# # model = NGBoost()
# model = XGBRegressor()

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# score = model.score(x_test, y_test)
# print('model.score : ', score)




# from sklearn.metrics import r2_score
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)

# print("r2_score : ", r2)

# print(model.feature_importances_)


# def NMAE(true, pred):
#     mae = np.mean(np.abs(true-pred))
#     score = mae / np.mean(np.abs(true))
#     return score

# NMAE(true, y_predict)

# # result['score_xgb']=NMAE(result['target'], result['prediction_xgb'])
# # print('XGB 의 score :', np.mean(result['score_xgb']))



# # score.to_csv('submission.csv', index=False)

# # sample_submission = pd.read_csv(path + 'sample_submission.csv')
# # sample_submission['target'] = pred
# # sample_submission

# # sample_submission.to_csv(PATH + "prediction5.csv", index=False)
 

 
# # model.score :  11.623987408689736

