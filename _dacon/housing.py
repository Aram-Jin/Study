import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path = "../_data/dacon/housing/"    

train = pd.read_csv(path + 'train.csv')
# print(train.shape)  # (1350, 15)
test_file = pd.read_csv(path + 'test.csv')
# print(test_file.shape)  # (1350, 14)
submit_file = pd.read_csv(path + 'sample_submission.csv')
# print(submit_file.shape)  # (1350, 2)



 

