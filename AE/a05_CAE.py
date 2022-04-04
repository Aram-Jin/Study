# 2번 카피해서 복붙
# CNN으로 딥하게 구성

# Conv2D
# Maxpool
# Conv2D
# Maxpool
# Conv2D    -> Encoder

# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D    -> Decoder

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D

#1. 데이터

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28).astype('float') / 255
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28 * 28).astype('float') / 255
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)

#2. 모델구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, UpSampling2D, MaxPool2D

# def autoencoder(hidden_layer_size):
#     model = Sequential([
#         Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'),
#         Dense(units=784, activation='sigmoid')
#         ])
#     return model