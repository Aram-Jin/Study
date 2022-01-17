
from hashlib import algorithms_available
from sklearn import datasets
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier')    # 모델의 갯수 :  41
allAlgorithms = all_estimators(type_filter='regressor')   # 모델의 갯수 :  55

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms))  

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name,'의 정답률 : ', r2)
    except:
        # continue
        print(name,'은 에러 터진 놈!!!!')

'''
ARDRegression 의 정답률 :  0.24926480107309645
AdaBoostRegressor 의 정답률 :  0.18037247522176192
BaggingRegressor 의 정답률 :  0.22146299079432308
BayesianRidge 의 정답률 :  0.24957561720517973
CCA 의 정답률 :  -0.1875185306363376
DecisionTreeRegressor 의 정답률 :  -0.20870428454848944
DummyRegressor 의 정답률 :  -0.0006494197429334214
ElasticNet 의 정답률 :  0.060591067855856995
ElasticNetCV 의 정답률 :  0.24125986412862332
ExtraTreeRegressor 의 정답률 :  -0.16202221424648933
ExtraTreesRegressor 의 정답률 :  0.1480182621234929
GammaRegressor 의 정답률 :  0.036102875217516206
GaussianProcessRegressor 의 정답률 :  -25.884086484688297
GradientBoostingRegressor 의 정답률 :  0.3252124015952429
HistGradientBoostingRegressor 의 정답률 :  0.3442698002031558
HuberRegressor 의 정답률 :  0.23406398732076872
IsotonicRegression 은 에러 터진 놈!!!!
KNeighborsRegressor 의 정답률 :  0.2637461691063886
KernelRidge 의 정답률 :  0.21879428218512786
Lars 의 정답률 :  0.2494896826312223
LarsCV 의 정답률 :  0.24955713996545115
Lasso 의 정답률 :  0.24789006679844872
LassoCV 의 정답률 :  0.24954124639126518
LassoLars 의 정답률 :  -0.0006494197429334214
LassoLarsCV 의 정답률 :  0.24955713996545115
LassoLarsIC 의 정답률 :  0.2495246227984803
LinearRegression 의 정답률 :  0.2494896826312223
LinearSVR 의 정답률 :  0.1903962187444579
MLPRegressor 의 정답률 :  0.253605181615436
MultiOutputRegressor 은 에러 터진 놈!!!!
MultiTaskElasticNet 은 에러 터진 놈!!!!
MultiTaskElasticNetCV 은 에러 터진 놈!!!!
MultiTaskLasso 은 에러 터진 놈!!!!
MultiTaskLassoCV 은 에러 터진 놈!!!!
NuSVR 의 정답률 :  0.20753327079523343
OrthogonalMatchingPursuit 의 정답률 :  0.13937599050521832
OrthogonalMatchingPursuitCV 의 정답률 :  0.24823627471987486
PLSCanonical 의 정답률 :  -0.6780629428885865
PLSRegression 의 정답률 :  0.24308584929802923
PassiveAggressiveRegressor 의 정답률 :  0.15864678939520138
PoissonRegressor 의 정답률 :  0.2418083239096911
QuantileRegressor 의 정답률 :  -0.055676209259269305
RANSACRegressor 의 정답률 :  0.0522195592228627
RadiusNeighborsRegressor 의 정답률 :  0.07024673124623959
RandomForestRegressor 의 정답률 :  0.2675623409753699
RegressorChain 은 에러 터진 놈!!!!
Ridge 의 정답률 :  0.24956613869112954
RidgeCV 의 정답률 :  0.24956613869109068
SGDRegressor 의 정답률 :  0.24960261900840175
SVR 의 정답률 :  0.20352287755602883
StackingRegressor 은 에러 터진 놈!!!!
TheilSenRegressor 의 정답률 :  0.24310707829989475
TransformedTargetRegressor 의 정답률 :  0.2494896826312223
TweedieRegressor 의 정답률 :  0.03552307392335319
VotingRegressor 은 에러 터진 놈!!!!
'''        