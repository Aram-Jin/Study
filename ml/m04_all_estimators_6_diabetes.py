from hashlib import algorithms_available
from sklearn import datasets
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
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
ARDRegression 의 정답률 :  0.498748289056254
AdaBoostRegressor 의 정답률 :  0.37218782815117346
BaggingRegressor 의 정답률 :  0.3109957537415179
BayesianRidge 의 정답률 :  0.5014366863847451
CCA 의 정답률 :  0.48696409064967594
DecisionTreeRegressor 의 정답률 :  -0.18551226170600055
DummyRegressor 의 정답률 :  -0.00015425885559339214
ElasticNet 의 정답률 :  0.11987522766332959
ElasticNetCV 의 정답률 :  0.48941369735908524
ExtraTreeRegressor 의 정답률 :  -0.40044833829077064
ExtraTreesRegressor 의 정답률 :  0.3934454885998395
GammaRegressor 의 정답률 :  0.07219655012236648
GaussianProcessRegressor 의 정답률 :  -7.547010959418039
GradientBoostingRegressor 의 정답률 :  0.39190149732199653
HistGradientBoostingRegressor 의 정답률 :  0.28899497703380905
HuberRegressor 의 정답률 :  0.5068530513878713
IsotonicRegression 은 에러 터진 놈!!!!
KNeighborsRegressor 의 정답률 :  0.3741821819765594
KernelRidge 의 정답률 :  0.4802268722469346
Lars 의 정답률 :  0.4919866521464151
LarsCV 의 정답률 :  0.5010892359535754
Lasso 의 정답률 :  0.46430753276688697
LassoCV 의 정답률 :  0.4992382182931273
LassoLars 의 정답률 :  0.3654388741895792
LassoLarsCV 의 정답률 :  0.4951942790678243
LassoLarsIC 의 정답률 :  0.49940515175310685
LinearRegression 의 정답률 :  0.5063891053505036
LinearSVR 의 정답률 :  0.1493389116863294
MLPRegressor 의 정답률 :  -0.6004045157319822
MultiOutputRegressor 은 에러 터진 놈!!!!
MultiTaskElasticNet 은 에러 터진 놈!!!!
MultiTaskElasticNetCV 은 에러 터진 놈!!!!
MultiTaskLasso 은 에러 터진 놈!!!!
MultiTaskLassoCV 은 에러 터진 놈!!!!
NuSVR 의 정답률 :  0.12527149380257419
OrthogonalMatchingPursuit 의 정답률 :  0.3293449115305741
OrthogonalMatchingPursuitCV 의 정답률 :  0.44354253337919725
PLSCanonical 의 정답률 :  -0.9750792277922931
PLSRegression 의 정답률 :  0.4766139460349792
PassiveAggressiveRegressor 의 정답률 :  0.4916064058466747
PoissonRegressor 의 정답률 :  0.4823231874912104
QuantileRegressor 의 정답률 :  -0.02193924207064546
RANSACRegressor 의 정답률 :  0.2531134890074581
RadiusNeighborsRegressor 의 정답률 :  0.14407236562185122
RandomForestRegressor 의 정답률 :  0.3772889393554766
RegressorChain 은 에러 터진 놈!!!!
Ridge 의 정답률 :  0.49950383964954104
RidgeCV 의 정답률 :  0.49950383964954104
SGDRegressor 의 정답률 :  0.4942357343636742
SVR 의 정답률 :  0.12343791188320263
StackingRegressor 은 에러 터진 놈!!!!
TheilSenRegressor 의 정답률 :  0.5101408244278444
TransformedTargetRegressor 의 정답률 :  0.5063891053505036
TweedieRegressor 의 정답률 :  0.07335459385974419
VotingRegressor 은 에러 터진 놈!!!!
'''
        