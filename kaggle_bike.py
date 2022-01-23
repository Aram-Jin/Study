from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from xgboost import XGBRegressor

#절대경로 & 상대경로 
path = 'D:\_data\kaggle\\bike/'

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['casual','registered','count'], axis=1)  
x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)

y = train['count']
y = np.log1p(y)
test_file['datetime'] = pd.to_datetime(test_file['datetime'])
test_file['year'] = test_file['datetime'].dt.year
test_file['month'] = test_file['datetime'].dt.month
test_file['day'] = test_file['datetime'].dt.day
test_file['hour'] = test_file['datetime'].dt.hour
test_file = test_file.drop('datetime', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

model = XGBRegressor(random_state = 66)

model.fit(x_train,y_train)
print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   

y_pred = model.predict(test_file)
y_pred_real = np.round(np.expm1(y_pred)).astype(int)
submit_file['count'] = y_pred_real
submit_file.to_csv( path + "submitfile1111.csv", index=False)
