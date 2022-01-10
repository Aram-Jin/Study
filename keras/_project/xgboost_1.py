from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance, plot_tree

# import graphviz
import matplotlib.pylab as plt
plt.style.use(['seaborn-whitegrid'])

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=123)
dtrain = xgb.DMatrix(data = x_train, label = y_train)
dtest = xgb.DMatrix(data = x_test, label = y_test)

params = {'max_depth': 3,
          'eta' : 0.1,
          'object': 'binary:logistic',
          'eval_metric': 'logloss',
          'early_stopping': 100}
num_rounds = 400

evals = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds,
                      early_stopping_rounds=100, evals=evals)