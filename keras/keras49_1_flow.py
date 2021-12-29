from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 변형하여 정의시킨다.
                                #    vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                #    rotation_range=5, 
                                   zoom_range=0.1, 
                                #    shear_range=0.7, 
                                   fill_mode='nearest')   # 1./255 minmax scale 지정 0~1사이로(사진에선 최대값이 255이므로), horizontal_flip은 좌우반전, vertical_flip은 상하반전, width_shift_range는 이미지 이동해도 인식(이미지증폭)

augument_size = 100  # 증폭하는 사이즈 정의

# print(x_train[0].shape)   # (28, 28)
# print(x_train[0].reshape(28*28).shape)   # (784,)
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape)   # (100, 28, 28, 1)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),  # x
    np.zeros(augument_size),                                                # y
    batch_size=augument_size,
    shuffle=False).next()       # flow는 x,y를 정의해주어야함, np.tile은 1차원어레이를 지정한 횟수만큼 반복하는 함수

print(type(x_data))   # <class 'tuple'>
# print(x_data)
print(x_data[0].shape, x_data[1].shape)   # (100, 28, 28, 1) (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()
