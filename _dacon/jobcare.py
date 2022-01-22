from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
path = "../_data/dacon/Jobcare_data/"    

train = pd.read_csv(path + 'train.csv')
print(train.shape)  # (501951, 35)
test_file = pd.read_csv(path + 'test.csv')
print(test_file.shape)  # (46404, 34)
submit_file = pd.read_csv(path + 'sample_submission.csv')
print(submit_file.shape)  # (46404, 2) 

x = train.drop(['id', 'contents_open_dt', 'target'], axis=1)  
print(x.shape)  # (501951, 33)

y = train['target']
print(y.shape)  # (501951,)

test_file = test_file.drop(['id', 'contents_open_dt'], axis=1)  
print(test_file.shape)   # (46404, 32)

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = [
    {'randomforestclassifier__max_depth' : [6, 8, 10], 'randomforestclassifier__min_samples_leaf' :[3, 5, 7]},
    {'randomforestclassifier__min_samples_leaf' :[3, 5, 7], 'randomforestclassifier__min_samples_split' : [3, 5, 10]}
]   

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

#2.모델
# model = SVC()
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())   

# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

from sklearn.metrics import accuracy_score
y_predict = model.predict(test_file)
acc = accuracy_score(y_test, y_predict)

print("걸린시간 : ", end - start)
print("model.score : ", result)
print("accuracy_score : ", acc)

submission = pd.read_csv(path + 'sample_submission.csv')
submission['target'] = y_predict

submission.to_csv('baseline.csv', index=False)





print(y_predict.shape)
print(y_predict)





# # from sklearn.model_selection import train_test_split, KFold, cross_val_score

# # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

# # n_splits = 5
# # kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)

# x = train.iloc[:, :-1]
# y = train.iloc[:, -1]

# model.fit(x,y)

# preds = model.predict(test)

# print(preds)

