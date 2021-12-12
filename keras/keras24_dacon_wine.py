import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from graphviz import Source

#1. 데이터
path = "../_data/dacon/wine/"
train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (3231, 14)

test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (3231, 13)

submit_file = pd.read_csv(path + 'sample_submission.csv')  
#print(submit_file.shape)  # (3231, 2)
#print(submit_file.columns)  # ['id', 'quality'], dtype='object

#print(type(train))  # <class 'pandas.core.frame.DataFrame'>
#print(train.columns)
#Index(['id', 'fixed acidity', 'volatile acidity', 'citric acid',
#       'residual sugar', 'chlorides', 'free sulfur dioxide',
#       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type',
#       'quality'],
#      dtype='object')

y = train['quality']
x = train.drop(['id', 'quality'], axis=1)  
x = x.drop(['sulphates'], axis=1) # 'citric acid', 'pH',
test_file = test_file.drop(['id', 'sulphates'], axis=1) # ,  'sulphates' 'citric acid', 'pH',
y = y.to_numpy()

le = LabelEncoder()
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

label2 = test_file['type']
le.fit(label2) 
test_file['type'] = le.transform(label2)

#print(test_file['type'])       # testfile의 type열의 값이 0,1로 바뀌어있는지 확인해봄.

#plt.figure(figsize=(10,10))
#sns.heatmap(data=x.columns.corr(), square=True, annot=True, cbar=True) 
#plt.show()

#print(np.unique(y))  # [4 5 6 7 8] 

#y = np.array(y).reshape(-1,1)
#enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
#enc.fit(y)
#y = enc.transform(y).toarray()
#x = x.to_numpy()

#y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=61)


#model2 = RandomForestClassifier(random_state=66)
#model2.fit(x_train, y_train)
#y_pred2 = model2.predict(x_test)

#print(y_pred2)


'''
rf = RandomForestClassifier(n_jobs = -1 , random_state =66)
score = cross_validate(rf, x_train, y_train, return_train_score=True, n_jobs= -1)
print(np.mean(score['train_score']), np.mean(score['test_score']))  # 1.0 0.46322102769406087

rf.fit(x_train, y_train)
print(rf.feature_importances_)   # [0.07706013 0.10137848 0.0758608  0.08726533 0.08503215 0.08534749
                                 #  0.08959861 0.10268387 0.07732534 0.08378797 0.13163588 0.00302395]

dt = DecisionTreeClassifier(random_state =66)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

decision_acc = accuracy_score(y_test, y_pred)
print('의사결정나무의 정확도: ',decision_acc)   #  0.5486862442040186

decision_report = classification_report(y_test, y_pred)
print(decision_report)

decision_matrix = confusion_matrix(y_test, y_pred)
print(decision_matrix)
#pred_train = dt.predict(x_train)
#pred_test = dt.predict(x_test)

#print(dt.feature_importances_)

rf = RandomForestClassifier(oob_score= True, n_jobs = -1 , random_state =66)
rf.fit(x_train, y_train)
print(rf.oob_score_)   # 0.8530959752321982

es = ExtraTreesClassifier(n_jobs = -1 , random_state =66)
score = cross_validate(es, x_train, y_train, return_train_score=True, n_jobs= -1)
print(np.mean(score['train_score']), np.mean(score['test_score']))   # 1.0 0.46361912044742326

#gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=66)
#score = cross_validate(gb, x_train, y_train, return_train_score=True, n_jobs= -1)
#print(np.mean(score['train_score']), np.mean(score['test_score'])) 

#print(y.shape)  # (3231, 5)
#print(x.shape)  # (3231, 9)
'''
#confusion_matrix(y_train, pred_train)
#confusion_matrix(y_test, pred_test)

#accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)
#recall_score(y_test, pred_test)
#precision_score(y_test, pred_test)

#eg = export_graphviz(dt, out_file=None, feature_names=x.columns, class_names=['White','Red'], rounded=True,filled=True)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

model2 = RandomForestClassifier(oob_score= True, n_estimators=15000, random_state=61)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(test_file)

print(model2.oob_score_)
print(y_pred2)

submit_file['quality'] = y_pred2
submit_file.to_csv(path+'subfile1.csv', index=False)
      
#submission['quality'] = y_pred2
#submission.to_csv("dacon_wine.csv", index=False)


'''
#2. 모델구성
model = Sequential() 
model.add(Dense(64, input_dim=12))
model.add(Dropout(0.2))  
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(5, activation='softmax'))
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  
filepath = './_dacon_wine/'                
filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
model_path = "".join([filepath, 'dacon_wine_', datetime_spot, '_', filename])

Es = EarlyStopping(monitor='val_accuracy', patience=800, mode='max', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath= model_path)
hist = model.fit(x_train, y_train, epochs=50000, batch_size=50, validation_split=0.2, callbacks=[Es, mcp])

model.save("./_save/keras24_3_save_model.h5") 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

##################################### 제출용 제작 #########################################
results = model.predict(test_file)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4

submit_file['quality'] = results_int

submit_file.to_csv(path+'subfile.csv', index=False)
      
acc = str(round(loss[1],4)).replace(".","_")
submit_file.to_csv(path +f"result/accuracy_{acc}.csv", index=False)


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

'''
Epoch 00565: val_accuracy did not improve from 0.54739
Epoch 00565: early stopping
21/21 [==============================] - 0s 1ms/step - loss: 1.0065 - accuracy: 0.5734
loss :  1.0065391063690186
accuracy :  0.5734157562255859

Epoch 00213: val_accuracy did not improve from 0.54932
Epoch 00213: early stopping
21/21 [==============================] - 0s 893us/step - loss: 1.0047 - accuracy: 0.5842
loss :  1.0046560764312744
accuracy :  0.5842349529266357

Epoch 00354: val_accuracy did not improve from 0.53578
Epoch 00354: early stopping
21/21 [==============================] - 0s 846us/step - loss: 0.9979 - accuracy: 0.5657
loss :  0.9978770613670349
accuracy :  0.5656877756118774

Epoch 00401: val_accuracy did not improve from 0.55899
Epoch 00401: early stopping
21/21 [==============================] - 0s 400us/step - loss: 0.9968 - accuracy: 0.5719
loss :  0.996820867061615
accuracy :  0.5718701481819153

loss :  0.9886926412582397
accuracy :  0.5765069723129272

loss :  0.995951235294342
accuracy :  0.5780525207519531
'''

'''

path = "../_data/dacon/wine/"
train = pd.read_csv(path + 'train.csv')

test_file = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

y = train['quality']
x = train.drop(['id', 'quality'], axis=1)  

test_file = test_file.drop(['id'], axis=1) # ,  'sulphates' 'citric acid', 'pH',
y = y.to_numpy()

le = LabelEncoder()
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

label2 = test_file['type']
le.fit(label2) 
test_file['type'] = le.transform(label2)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

model2 = RandomForestClassifier(oob_score= True, n_estimators=500, random_state=66)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(test_file)

print(model2.oob_score_)
print(y_pred2)

submit_file['quality'] = y_pred2
submit_file.to_csv(path+'subfile.csv', index=False)


'''