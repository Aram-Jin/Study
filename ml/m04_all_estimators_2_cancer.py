from hashlib import algorithms_available
from sklearn import datasets
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')    # 모델의 갯수 :  41
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

'''
AdaBoostClassifier 의 정답률 :  0.9473684210526315
BaggingClassifier 의 정답률 :  0.9649122807017544
BernoulliNB 의 정답률 :  0.6403508771929824
CalibratedClassifierCV 의 정답률 :  0.9649122807017544
ComplementNB 의 정답률 :  0.7807017543859649
DecisionTreeClassifier 의 정답률 :  0.9210526315789473
DummyClassifier 의 정답률 :  0.6403508771929824
ExtraTreeClassifier 의 정답률 :  0.9298245614035088
ExtraTreesClassifier 의 정답률 :  0.956140350877193
GaussianNB 의 정답률 :  0.9210526315789473
GaussianProcessClassifier 의 정답률 :  0.9649122807017544
GradientBoostingClassifier 의 정답률 :  0.956140350877193
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
KNeighborsClassifier 의 정답률 :  0.956140350877193
LabelPropagation 의 정답률 :  0.9473684210526315
LabelSpreading 의 정답률 :  0.9473684210526315
LinearDiscriminantAnalysis 의 정답률 :  0.9473684210526315
LinearSVC 의 정답률 :  0.9736842105263158
LogisticRegression 의 정답률 :  0.9649122807017544
LogisticRegressionCV 의 정답률 :  0.9736842105263158
MLPClassifier 의 정답률 :  0.9824561403508771
MultinomialNB 의 정답률 :  0.8508771929824561
NearestCentroid 의 정답률 :  0.9298245614035088
NuSVC 의 정답률 :  0.9473684210526315
PassiveAggressiveClassifier 의 정답률 :  0.9736842105263158
Perceptron 의 정답률 :  0.9736842105263158
QuadraticDiscriminantAnalysis 의 정답률 :  0.9385964912280702
RandomForestClassifier 의 정답률 :  0.956140350877193
RidgeClassifier 의 정답률 :  0.9473684210526315
RidgeClassifierCV 의 정답률 :  0.9473684210526315
SGDClassifier 의 정답률 :  0.9385964912280702
SVC 의 정답률 :  0.9736842105263158
'''