import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
 

data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")
data3 = pd.read_csv("./평가종목data.csv")

# print(type(data1))
# print(type(data2))

dataset = pd.concat([data1,data2],ignore_index=True).astype('float')
pre_data = data3.astype('float')
# print(type(pre_data))
del dataset['Unnamed: 0']
del pre_data['Unnamed: 0']

x = dataset.drop(['Target'], axis=1)
y = dataset['Target']


#print(dft_X.isnull().sum()) #dft_X-> 통합 XGBOOST CSV

# MAX TEMP(℃)    0
# MIN TEMP(℃)    0
# PM10           0
#  PM2.5         0
# SO2            0
# O3             0
# NO2            0
# CO             0
# NUM            0
# Domestic       0
# Foreign        0
# MEN            0
# WOMEN          0
# INSPECTION     0
# dtype: int64

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
xgb_model=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsampel=0.75, colsample_bytree=1, max_depth=7)

#print(len(dft_x_train), len(dft_x_test))
xgb_model.fit(x_train, y_train)

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1,
             missing=None, n_estimators=400, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=True, subsample=0.75)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)

plt.show()

pred=xgb_model.predict(x_test)
print(pred)

# [ 6.3407645   3.2552927   9.489924    5.1280856  17.028868   16.949871
#  22.001284   11.476369    6.434142   15.403754   10.20475     2.3318226
#  14.79987    25.001297    4.559653   16.671524    2.4657805  -5.389366
#  17.568739   26.583458    2.4746122  10.327534   10.786948   23.787401
#  -5.5242767  20.845575   27.584475   25.489971   20.703094   22.209776
#  24.77626    20.93491    10.283435    4.015735   -3.460021   14.554295
#  14.196549   22.545605   24.32385     8.224135   26.482687   25.645407
#  -0.52771086 10.208914   22.10409    11.229499    9.8236885  10.112332
#  10.286304    6.247892   10.669618    1.5466396  22.517584    8.30513
#   0.8229522   7.0456996  23.087389   15.715689   15.391215   25.215046
#  11.559614    2.8598976  17.059185   10.007217    7.795856   20.898386
#  14.738717   16.610312   24.161892   13.9859085  21.164238    7.7438245
#  16.17835    25.10558    14.776901    3.5199063   1.6414483  11.694139
#  -0.1488446  24.528368    7.193819   20.638857   19.825935    2.3564265
#  19.037474   11.845727   12.763272   21.32217     1.0469593  19.300903
#   2.2801085   4.348495   15.583948    5.196142   -1.853693   14.854492
#  12.476118   19.52385    27.379192   17.442575   19.352642    3.4604838
#  15.378529   -0.98452127 26.473394   22.605728   18.538467   18.207851
#   6.7919464   3.7254157  16.735044   14.472764   14.364736    6.7923427
#  14.494831   25.561455    3.5038414   4.2043233  14.651863   25.570038
#   8.022747   21.61166    19.60632    11.112737    3.7776124  10.229921
#  -2.2019467  -1.9245603   4.267021   15.300712    3.0718563  13.999785
#   8.815793   15.544661    2.64125    -5.8582187  11.387004  ]

r_sq=xgb_model.score(x_train, y_train)
print(r_sq)
print(explained_variance_score(pred, y_test))



# kfx=get_clf_eval(dft_y_test, pred)
# print(kfx)

# r_sq=xgb_model.score(dft_x_train, dft_y_train)

# print(explained_variance_score(predictions, dft_y_test))

