import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#1 데이터
path = "D:\\_data\\dacon\\wine\\" 
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv")

y = train['quality']
x = train.drop(['id','quality'], axis =1)
test_file =test_file.drop(['id'], axis=1)

x = x.drop(['citric acid'],axis =1)
test_file =test_file.drop(['citric acid'],axis =1)

#x = x.drop(['citric acid','pH','sulphates','total sulfur dioxide'],axis =1)
#test_file =test_file.drop(['citric acid','pH','sulphates','total sulfur dioxide'],axis =1)

le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

y = y.to_numpy()
x = x.to_numpy()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model1 = RandomForestClassifier(oob_score= True, bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators= 1000, n_jobs=None, verbose=0, warm_start=False, random_state=61)

model2 = GradientBoostingClassifier(n_estimators = 1000,random_state=66)

model3 = ExtraTreesClassifier(oob_score= True, bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators= 1000, n_jobs=None, verbose=0, warm_start=False, random_state=61)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
model4 = HistGradientBoostingClassifier(max_iter = 750, random_state =66)

from lightgbm import LGBMClassifier
model5 = LGBMClassifier(n_estimators= 750,random_state =66)


voting_model = VotingClassifier(estimators=[ ('RandomForestClassifier', model1), ('GradientBoostingClassifier', model2)
                                            ,('ExtraTreesClassifier', model3),('HistGradientBoostingClassifier', model4),('LGBMClassifier', model5) ], voting='hard')

classifiers = [model1, model2,model3,model4,model5]

for classifier in classifiers:
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    class_name = classifier.__class__.__name__
    print("============== " + class_name + " ==================")
    num = accuracy_score(y_test, pred)
    print('{0} 정확도: {1}'.format(class_name, num))
    y_pred_ = classifier.predict(test_file)
    submission['quality'] = y_pred_
    submission.to_csv(str(num) +"_" + class_name + "_dacon_wine_vote.csv", index=False)
   

voting_model.fit(x_train, y_train)
pred = voting_model.predict(x_test)

print('===================== 보팅 분류기 ========================')
num = str(accuracy_score(y_test, pred))
print('{0} 정확도: {1}'.format(num))

y_pred_ = voting_model.predict(test_file)

submission['quality'] = y_pred_
submission.to_csv(num + "_dacon_wine.csv", index=False)

