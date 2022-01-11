from seaborn.matrix import heatmap
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
import seaborn as sns
import matplotlib.pyplot as plt
 

data1 = pd.read_csv("./관리종목data.csv")
# data2 = pd.read_csv("./안전종목data.csv")

print(data1.info())

data1=data1.drop(['Target','CR1','QR1','DR1','RR1','NDR1','ICR1','SGR1','SAEGR1','EBITDA1','GPM1','OPP1',
                       'CR2','QR2','DR2','RR2','NDR2','ICR2','SGR2','SAEGR2','EBITDA2','GPM2','OPP2',
                       'CR3','QR3','DR3','RR3','NDR3','ICR3','SGR3','SAEGR3','EBITDA3','GPM3','OPP3',
                       'CR4','QR4','DR4','RR4','NDR4','ICR4','SGR4','SAEGR4','EBITDA4','GPM4','OPP4'], axis=1)

print(data1.info())

colormap=plt.cm.PuBu
plt.figure(figsize=(20,20))
plt.title("2021 Financial Ratio of Fteatures", y=1.00, size=15)
sns.heatmap(data1.astype(float).corr(), linewidths=0.08, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True, annot_kws={"size":6})

plt.show()


# print(data2.info())

# data2=data2.drop(['Target','CR1','QR1','DR1','RR1','NDR1','ICR1','SGR1','SAEGR1','EBITDA1','GPM1','OPP1',
#                        'CR2','QR2','DR2','RR2','NDR2','ICR2','SGR2','SAEGR2','EBITDA2','GPM2','OPP2',
#                        'CR3','QR3','DR3','RR3','NDR3','ICR3','SGR3','SAEGR3','EBITDA3','GPM3','OPP3',
#                        'CR4','QR4','DR4','RR4','NDR4','ICR4','SGR4','SAEGR4','EBITDA4','GPM4','OPP4'], axis=1)

# print(data2.info())

# colormap=plt.cm.PuBu
# plt.figure(figsize=(20,20))
# plt.title("2021 Financial Ratio of Fteatures", y=1.00, size=15)
# sns.heatmap(data2.astype(float).corr(), linewidths=0.08, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True, annot_kws={"size":6})

# plt.show()