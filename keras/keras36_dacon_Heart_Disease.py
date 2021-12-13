import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path = "../_data/dacon/Heart_Disease/"
train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (151, 15)

test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (152, 14)

submit_file = pd.read_csv(path + 'sample_submission.csv')  
#print(submit_file.shape)  # (152, 2) 
#print(submit_file.columns)  # ['id', 'target'], dtype='object

#print(type(train))  # <class 'pandas.core.frame.DataFrame'>
#print(train.columns)
#Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
#      dtype='object')

#print(train.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 151 entries, 0 to 150
Data columns (total 15 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   id        151 non-null    int64
 1   age       151 non-null    int64
 2   sex       151 non-null    int64
 3   cp        151 non-null    int64
 4   trestbps  151 non-null    int64
 5   chol      151 non-null    int64
 6   fbs       151 non-null    int64
 7   restecg   151 non-null    int64
 8   thalach   151 non-null    int64
 9   exang     151 non-null    int64
 10  oldpeak   151 non-null    float64
 11  slope     151 non-null    int64
 12  ca        151 non-null    int64
 13  thal      151 non-null    int64
 14  target    151 non-null    int64
dtypes: float64(1), int64(14)
memory usage: 17.8 KB
None
'''

y = train['target']
x = train.drop(['id', 'target'], axis=1)  
#x = x.drop(['  '], axis=1) 
test_file = test_file.drop(['id'], axis=1) 

#print(x.shape, y.shape)   # (151, 13) (151,)
#print(np.unique(y))   # [0 1]
#print(y.shape)   # (151, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=61)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2. 모델구성
model = Sequential() 
model.add(Dense(64, input_dim=13))
model.add(Dropout(0.2))  
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(1, activation='sigmoid'))
#model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')

#date = datetime.datetime.now()
#datetime_spot = date.strftime("%m%d_%H%M")  
#filepath = './_dacon_wine/'                
#filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
#model_path = "".join([filepath, 'dacon_wine_', datetime_spot, '_', filename])

Es = EarlyStopping(monitor='val_loss', patience=1000, mode='min', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2, callbacks=[Es])   #callbacks=[Es, mcp]

#model.save("./_save/keras24_3_save_model.h5") 

#4. 결과예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
#print(y_predict)

y_predict2 = y_predict.round(0).astype(int)
#print(y_predict2)
#print(y_predict2.shape)  # (152, 1)

result = model.predict(test_file)
test_file = result.round(0).astype(int)
#print(test_file)


#submit_file['target'] = test_file
#submit_file.to_csv(path+'subfile.csv', index=False)

