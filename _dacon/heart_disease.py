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

#1. 데이터
path = "../_data/dacon/Heart_Disease/"
train = pd.read_csv(path + 'train.csv')
# print(train.shape)  # (151, 15)

test_file = pd.read_csv(path + 'test.csv')
# print(test_file.shape)  # (152, 14)

submit_file = pd.read_csv(path + 'sample_submission.csv')  
# print(submit_file.shape)  # (152, 2) 
# print(submit_file.columns)  # ['id', 'target'], dtype='object

# print(type(train))  # <class 'pandas.core.frame.DataFrame'>
# print(train.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
#      dtype='object')
# print(train.info())


y = train['target']
x = train.drop(['id', 'target'], axis=1)  
#x = x.drop(['  '], axis=1) 
test_file = test_file.drop(['id'], axis=1) 

# print(x.shape, y.shape)   # (151, 9) (151,)
# print(np.unique(y))   # [0 1]
# print(y.shape)   # (151,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=61)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
#scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_file = scaler.transform(test_file)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train.shape, x_test.shape) # (120, 13) (31, 13)

#2. 모델구성
model = Sequential() 
model.add(Dense(162, input_dim=13))
# model.add(Dropout(0.2))  
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))  
model.add(Dense(32))
model.add(Dropout(0.1))  
model.add(Dense(16, activation='relu')) 
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')

#date = datetime.datetime.now()
#datetime_spot = date.strftime("%m%d_%H%M")  
#filepath = './_dacon_wine/'                
#filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
#model_path = "".join([filepath, 'dacon_', datetime_spot, '_', filename])

Es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='f1_score', mode='min', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[Es])   

model.save("./_save/heart_disease_save_model.h5") 

#4. 결과예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
#print(y_predict)

y_predict2 = y_predict.round(0).astype(int)
#print(y_predict2)
#print(y_predict2.shape)  # (152, 1)

f1 = f1_score(y_test, y_predict2)  #, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')/// average : binary, macro, micro, weighted, None  / zero_division : 1
#print(f1)   # 0.8
confusion = confusion_matrix(y_test, y_predict2) #, labels=None, sample_weight=None, normalize=None)
accuracy = accuracy_score(y_test, y_predict2) #, normalize=True, sample_weight=None)
precision = precision_score(y_test, y_predict2) #, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
recall = recall_score(y_test, y_predict2) #, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

print('accuracy_score :', accuracy)	
print('recall_score: ', recall)	
print('precision_score: ', precision)	
print('f1_score: ', f1)	

print('오차행렬')
print(confusion)
print('정확도 : {:.4f}\n정밀도 : {:.4f}\n재현율 : {:.4f}'.format(accuracy, precision, recall))

# 임계값을 낮출 수록 TPR(재현율)이 올라가는 것을 확인했었다.
print(classification_report(y_test, y_predict2))

'''
              precision    recall  f1-score   support
           0       0.64      0.70      0.67        10
           1       0.85      0.81      0.83        21
    accuracy                           0.77        31
   macro avg       0.74      0.75      0.75        31
weighted avg       0.78      0.77      0.78        31
'''

roc_curve = roc_curve(y_test, y_predict2, pos_label=None, sample_weight=None, drop_intermediate=True)
roc_auc_score = roc_auc_score(y_test, y_predict2, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)

print("ROC curve: ", roc_curve)   # (array([0. , 0.1, 1. ]), array([0.        , 0.80952381, 1.        ]), array([2, 1, 0]))
print("ROC | AUC Score: ", roc_auc_score)   # 0.8547619047619047

result = model.predict(test_file)
test_file = result.round(0).astype(int)
#print(test_file)


submit_file['target'] = test_file
submit_file.to_csv(path+'subfile.csv', index=False)


