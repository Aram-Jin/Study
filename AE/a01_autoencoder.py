# 앞뒤가 똑같은 autoencoder

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

#2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape = (784,))
#encoded = Dense(16, activation = 'relu')(input_img) 
#encoded = Dense(64, activation = 'relu')(input_img) 
encoded = Dense(154, activation = 'relu')(input_img)   # PCA 0.95 지점
# encoded = Dense(486, activation = 'relu')(input_img)    # PCA 0.999 지점
# encoded = Dense(1024, activation = 'relu')(input_img) 

decoded = Dense(784, activation = 'sigmoid')(encoded)  # autoencoder는 input, output(앞,뒤)이 같기 때문에 똑같이 784
                                  # linear
                                                       # why 중간에 layer를 줄였다가 다시 복구를 해줄까?
                                                       # layers를 축소 시킴으로써 특성이 작은 것을 사라지게 해준다.
                                                       # ex) 얼굴의 기미, 주근깨 같은고를 없애준당.

autoencoder = Model(input_img, decoded)

#3. 컴파일,훈련
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
#autoencoder.compile(optimizer = 'adam', loss = 'mse')

autoencoder.fit(x_train, x_train, epochs = 30, batch_size = 128, validation_split = 0.2)

#4. 평가,예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
 