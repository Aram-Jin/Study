from numpy.random import rand
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

augment_size = 40000  
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # randint : 랜덤한 정수값을 생성하겠다. (여기선 0~60000개중 40000개 추출)
print(x_train.shape[0])   # 60000
print(randidx)   # [ 6446 39645 27996 ...  8500 42272 46455]  => 형태는 list형태로 되어있음
print(np.min(randidx), np.max(randidx))   # 0 59999
print(randidx.shape)   # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)   # (40000, 28, 28)
print(y_augmented.shape)   # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                    x_augmented.shape[1],
                                    x_augmented.shape[2], 1)
                                    
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented, #np.zeros(augument_size),
                                  batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented)
print(x_augmented.shape)   # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))   # concatenate를 사용할때는 (())괄호 두개로 사용해주어야함
y_train = np.concatenate((y_train, y_augmented))

print(x_train)
print(x_train.shape, y_train.shape)   # (100000, 28, 28, 1) (100000,) -> 기존 60000개에서 랜덤하게 추출한 40000개를 붙여줌

#1. x_augmented 10개와 x_train 10개를 비교하는 이미지 출력 할 것
# subplot(2, 10, ? )사용

# print(x_augmented[randidx][:10])

import matplotlib.pyplot as plt
plt.figure(figsize=(2,10))
for i in range(99):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[0][i], cmap='gray')
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(2,10))
for i in range(99):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_augmented[0][i], cmap='rainbow')
plt.show()

