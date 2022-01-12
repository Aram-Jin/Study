import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import set_option 
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV 
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier 
from sklearn.metrics import mean_squared_error 
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier 
from catboost import CatBoostClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing 
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler 
import seaborn as sns 

data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")
data3 = pd.read_csv("./평가종목data.csv")

dataset = pd.concat([data1,data2],ignore_index=True).astype('float')
pre_data = data3.astype('float')
# print(type(pre_data))
del dataset['Unnamed: 0']
del pre_data['Unnamed: 0']

# pre_data = np.array(pre_data)
# pre_data = pre_data.reshape(len(pre_data),11,5)
# print(pre_data.shape)

# print(dataset)

# dataset.to_csv('data_reset.csv', index=True, encoding='utf-8-sig')

# print(dataset.info())
# print(dataset.feature_names)
# print(dataset.DESCR)
# print(np.min(dataset), np.max(dataset))
# print(dataset.head)
# for col in dataset.columns:
#     print(col)
# print(dataset.index)
# np.array(dataset)

x = dataset.drop(['Target'], axis=1)
y = dataset['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)
 
model1 = RandomForestClassifier(n_estimators = 100, max_depth=12, min_samples_split=8, min_samples_leaf =8, n_jobs = -1) 
model2 = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model3 = ExtraTreesClassifier(n_estimators=100, max_depth=16, random_state=7) 
model4 = AdaBoostClassifier(n_estimators=30, random_state=10, learning_rate=0.1)  #  trade-off 관계입니다. n_estimators(또는 learning_rate)를 늘리고, learning_rate(또는 n_estimators)을 줄인다면 서로 효과가 상쇄된다.→ 때문에 이 두 파라미터를 잘 조정하는 것이 알고리즘의 핵심입니다.
model5 = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3, min_samples_split=40, min_samples_leaf =30) 
model6 = LGBMClassifier(n_estimators = 100, learning_rate = 0.1, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model7 = CatBoostClassifier(n_estimators=100, max_depth=16, random_state=7) 
 
from sklearn.ensemble import VotingClassifier 
voting_model = VotingClassifier(estimators=[('RandomForestClassifier', model1), 
                                            ('GradientBoostingClassifier', model2), 
                                            ('ExtraTreesClassifier', model3), 
                                            ('AdaBoostClassifier', model4), 
                                            ('XGBClassifier', model5), 
                                            ('LGBMClassifier', model6), 
                                            ('CatBoostClassifier', model7)], voting='hard') 
 
classifiers = [model1,model2,model3,model4,model5,model6,model7] 
from sklearn.metrics import accuracy_score 
 
for classifier in classifiers: 
    classifier.fit(x_train, y_train) 
    y_predict = classifier.predict(pre_data) 
    accuracy = accuracy_score(y_train,y_predict) 
           
    class_name = classifier.__class__.__name__ 
    print("============== " + class_name + " ==================") 
     
    print('accuracy 스코어 : ', accuracy) 
    print('예측값 : ', y_predict) 
 


'''
1. Boosting Algorithm
부스팅 알고리즘은 여러 개의 약한 학습기(weak learner)를 순차적으로 학습-예측하면서 잘못 예측한 데이터에 가중치를 부여해 오류를 개선해나가며 학습하는 방식

부스팅 알고리즘은 대표적으로 아래와 같은 알고리즘들이 있음

AdaBoost
Gradient Booting Machine(GBM)
XGBoost
LightGBM
CatBoost


2. AdaBoost
Adaptive Boost의 줄임말로서 약한 학습기(weak learner)의 오류 데이터에 가중치를 부여하면서 부스팅을 수행하는 대표적인 알고리즘
속도나 성능적인 측면에서 decision tree를 약한 학습기로 사용함

* n_estimators : 생성할 약한 학습기의 갯수를 지정. Default = 50
* learning_rate : 학습을 진행할 때마다 적용하는 학습률(0~1). Weak learner가 순차적으로 오류 값을 보정해나갈 때 적용하는 계수. Default = 1.0

n_estimators를 늘린다면,
생성하는 weak learner의 수는 늘어남
이 여러 학습기들의 decision boundary가 많아지면서 모델이 복잡해짐
learning_rate을 줄인다면
가중치 갱신의 변동폭이 감소해서, 여러 학습기들의 decision boundary 차이가 줄어듦
위의 두 가지는 trade-off 관계입니다.

n_estimators(또는 learning_rate)를 늘리고, learning_rate(또는 n_estimators)을 줄인다면 서로 효과가 상쇄됩다.
→ 때문에 이 두 파라미터를 잘 조정하는 것이 알고리즘의 핵심입니다.


3. Gradient Boost Machine(GBM)
AdaBoost와 유사하지만, 가중치 업데이트를 경사하강법(Gradient Descent)를 이용하여 최적화된 결과를 얻는 알고리즘입니다.
GBM은 예측 성능이 높지만 Greedy Algorithm으로 과적합이 빠르게되고, 시간이 오래 걸린다는 단점이 있습니다

GBM의 하이퍼파라미터
* Tree에 관한 하이퍼 파라미터
    (1) max_depth : 
    - 트리의 최대 깊이
    - default = 3
    - 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요
    
    (2) min_samples_split :
    - 노드를 분할하기 위한 최소한의 샘플 데이터수
    → 과적합을 제어하는데 사용
    - Default = 2 → 작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가
    
    (3) min_samples_leaf :
    - 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수
    - min_samples_split과 함께 과적합 제어 용도
    - default = 1
    - 불균형 데이터의 경우 특정 클래스의 데이터가 극도로 작을 수 있으므로 작게 설정 필요

    (4) max_features :
    - 최적의 분할을 위해 고려할 최대 feature 개수
    - Default = 'none' → 모든 피처 사용
    - int형으로 지정 →피처 갯수 / float형으로 지정 →비중
    - sqrt 또는 auto : 전체 피처 중 √(피처개수) 만큼 선정
    - log : 전체 피처 중 log2(전체 피처 개수) 만큼 선정
    
    (5) max_leaf_nodes :
    - 리프노드의 최대 개수
    - default = None → 제한없음
   
* Boosting에 관한 하이퍼파라미터    
    (1) loss :
    - 경사하강법에서 사용할 cost function 지정
    - 특별한 이유가 없으면 default 값인 deviance 적용
    
    (2) n_estimators :
    - 생성할 트리의 갯수를 지정
    - Default = 100
    - 많을소록 성능은 좋아지지만 시간이 오래 걸림
    
    (3) learning_rate :
    - 학습을 진행할 때마다 적용하는 학습률(0~1)
    - Weak learner가 순차적으로 오류 값을 보정해나갈 때 적용하는 계수
    - Default = 0.1
    - 낮은 만큼 최소 오류 값을 찾아 예측성능이 높아질 수 있음
    - 하지만 많은 수의 트리가 필요하고 시간이 많이 소요
    
    (4) subsample :
    - 개별 트리가 학습에 사용하는 데이터 샘플링 비율(0~1)
    - default=1 (전체 데이터 학습)
    - 이 값을 조절하여 트리 간의 상관도를 줄일 수 있음
    
    
4. XGBoost(eXtra Gradient Boost) 
트리 기반의 알고리즘의 앙상블 학습에서 각광받는 알고리즘 중 하나입니다.
GBM에 기반하고 있지만, GBM의 단점인 느린 수행시간, 과적합 규제 등을 해결한 알고리즘입니다.   
    
    XGBoost의 주요장점
    (1) 뛰어난 예측 성능
    (2) GBM 대비 빠른 수행 시간
    (3) 과적합 규제(Overfitting Regularization)
    (4) Tree pruning(트리 가지치기) : 긍정 이득이 없는 분할을 가지치기해서 분할 수를 줄임
    (5) 자체 내장된 교차 검증

    반복 수행시마다 내부적으로 교차검증을 수행해 최적회된 반복 수행횟수를 가질 수 있음
    지정된 반복횟수가 아니라 교차검증을 통해 평가 데이트세트의 평가 값이 최적화되면 반복을 중간에 멈출 수 있는 기능이 있음
    (6) 결손값 자체 처리
        
XGBoost의 하이퍼 파라미터
* 일반 파라미터
    booster : 
    - gbtree(tree based model) 또는 gblinear(linear model) 중 선택
    - Default = 'gbtree'

5. LightGBM 
    
    LightGBM의 장점
    (1) XGBoost 대비 더 빠른 학습과 예측 수행 시간
    (2) 더 작은 메무리 사용량
    (3) 카테고리형 피처의 자동 변환과 최적 분할
    : 원-핫인코딩 등을 사용하지 않고도 카테고리형 피처를 최적으로 변환하고 이에 따른 노드분할 수행

    LightGBM의 단점
    적은 데이터 세트에 적용할 경우 과적합이 발생하기 쉽습니다.
    (공식 문서상 대략 10,000건 이하의 데이터 세트)

    기존 GBM과의 차이점
    일반적인 균형트리분할 (Level Wise) 방식과 달리 리프중심 트리분할(Leaf Wise) 방식을 사용합니다.

    균형트리분할은 최대한 균형 잡힌 트리를 유지하며 분할하여 트리의 깊이를 최소화하여
    오버피팅에 강한구조이지만 균형을 맞추기 위한 시간이 필요합니다.
    리프중심 트리분할의 경우 최대 손실 값을 가지는 리프노드를 지속적으로 분할하면서
    트리가 깊어지고 비대칭적으로 생성합니다. 이로써 예측 오류 손실을 최소화하고자 합니다.

    .....
    ...
    
    
    
    
    
    
    
    
    
    
    




'''