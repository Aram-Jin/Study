from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('../_data/dacon/Jobcare_data/train.csv')
test = pd.read_csv('../_data/dacon/Jobcare_data/test.csv')

print(train.head())
print(test.head())

train = train.drop(['id', 'contents_open_dt'], axis=1) 
test = test.drop(['id', 'contents_open_dt'], axis=1)

# from sklearn.model_selection import train_test_split, KFold, cross_val_score

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = RandomForestClassifier(n_estimators=300, max_depth=60, n_jobs=-1)

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)

preds = model.predict(test)

print(preds)

# submission = pd.read_csv('sample_submission.csv')
# submission['target'] = preds

# submission.to_csv('baseline.csv', index=False)