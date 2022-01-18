from hashlib import algorithms_available
from sklearn import datasets
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')   # # 모델의 갯수 :  41
# allAlgorithms = all_estimators(type_filter='regressor')   # 모델의 갯수 :  55

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms))   # 모델의 갯수 :  41

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name,'의 정답률 : ', acc)
    except:
        continue
        # print(name,'은 에러 터진 놈!!!!')
        
'''
모델의 갯수 :  41
AdaBoostClassifier 의 정답률 :  0.8888888888888888
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  0.9722222222222222
CategoricalNB 의 정답률 :  0.5
ComplementNB 의 정답률 :  0.8611111111111112
DecisionTreeClassifier 의 정답률 :  0.9722222222222222
DummyClassifier 의 정답률 :  0.4166666666666667
ExtraTreeClassifier 의 정답률 :  0.9166666666666666
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  1.0
GradientBoostingClassifier 의 정답률 :  0.9722222222222222
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9722222222222222
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  0.9722222222222222
MLPClassifier 의 정답률 :  1.0
MultinomialNB 의 정답률 :  0.9444444444444444
NearestCentroid 의 정답률 :  1.0
NuSVC 의 정답률 :  1.0
PassiveAggressiveClassifier 의 정답률 :  0.9722222222222222
Perceptron 의 정답률 :  0.9722222222222222
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RadiusNeighborsClassifier 의 정답률 :  0.9722222222222222
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  0.9722222222222222
SGDClassifier 의 정답률 :  0.9722222222222222
SVC 의 정답률 :  1.0
'''