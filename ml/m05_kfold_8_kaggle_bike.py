import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # Classifier :분류모델
from sklearn.linear_model import LogisticRegression,LinearRegression        # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y.shape)  # (10886,)

test_file = test_file.drop(['datetime'], axis=1)  

from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print(scores)
print("r2 : ", scores, "\n cross_val_score : ", np.mean(scores),4)

'''
Perceptron
r2 :  [0.01262916 0.00114811 0.00401837 0.00459506 0.01263642] 
 cross_val_score :  0.0070054226723493835 4
 
LinearSVC
r2 :  [0.00172216 0.00057405 0.00057405 0.00057438 0.00057438] 
 cross_val_score :  0.0008038058283671116 4
 
SVC
r2 :  [0.02353617 0.01549943 0.02238806 0.01033889 0.01608271] 
 cross_val_score :  0.01756904955186951 4

KNeighborsRegressor 
r2 :  [0.16794866 0.18517249 0.23680429 0.16578537 0.23294712] 
 cross_val_score :  0.19773158431852578 4 
 
LinearRegression
r2 :  [0.2369219  0.26392691 0.29226917 0.24560392 0.26296619] 
 cross_val_score :  0.2603376167711354 4
 
DecisionTreeRegressor
r2 :  [-0.30921532 -0.01996092 -0.10513734 -0.30745117 -0.22607942] 
 cross_val_score :  -0.1935688341594522 4
 
RandomForestRegressor
r2 :  [0.22385691 0.30075949 0.33904989 0.27608804 0.31514546] 
 cross_val_score :  0.2909799607731679 4
'''
