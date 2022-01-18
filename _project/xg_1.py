import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
 
data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")


dataset = pd.concat([data1,data2],ignore_index=True)
del dataset['Unnamed: 0']

x = dataset.drop(['Target'], axis=1)
y = dataset['Target']

df4_x_train, df4_x_test, df4_y_train, df4_y_test=train_test_split(x, y, test_size=0.2)
xgb_model=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsampel=0.75, colsample_bytree=1, max_depth=7)

xgb_model.fit(df4_x_train, df4_y_train)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
plt.show()

y_pred = xgb_model.predict(df4_x_test)
print(y_pred)