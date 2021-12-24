from numpy.core.fromnumeric import shape
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (150, 4) (150,)

print(np.unique(y))   # [0 1 2]

#y = to_categorical(y)
#print(y.shape)   # (150, 3)

x_train, y_train, x_test, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,1)
x_test = scaler.transform(x_test).reshape(len(x_test),2,2,1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(4, kernel_size=(2,2), padding='same', strides=1, input_shape=(2,2,1)))
model.add(Conv2D(4, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(9, activation='relu'))
model.add(Dense(6))
model.add(Dense(3, activation='softmax'))
model.summary()

#3. 컴파일, 훈련