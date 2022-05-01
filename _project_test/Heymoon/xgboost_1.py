from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance, plot_tree
import graphviz
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

predicts = xgb_model.predict(dtest)
print(np.round(predicts[:10], 3))
preds = [1 if x > 0.5 else 0 for x in predicts]
print(preds[:10])

print("정확도: {}".format(accuracy_score(y_test, preds)))
print("정밀도: {}".format(precision_score(y_test, preds)))
print("재현율: {}".format(recall_score(y_test, preds)))

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax);
# print(plot_importance(xgb_model, ax=ax))

dot_data = xgb.to_graphviz(xgb_model)
graph = graphviz.Source(dot_data)
