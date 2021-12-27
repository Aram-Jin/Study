import numpy as np
from numpy.core.defchararray import array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import sequential


#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255, )                      # ImageDataGenerator 정의하기
                                #    horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
                                #    vertical_flip=True, 
                                #    width_shift_range=0.1, 
                                #    height_shift_range=0.1, 
                                #    rotation_range=5, 
                                #    zoom_range=1.2, 
                                #    shear_range=0.7, 
                                #    fill_mode='nearest')   # 1./255 minmax scale 지정 0~1사이로(사진에선 최대값이 255이므로), horizontal_flip은 상하반전, vertical_flip은 좌우반전, width_shift_range는 이미지 이동해도 인식(이미지증폭)

test_datagen = ImageDataGenerator(rescale=1./255)   #test데이터는 평가만 할 것이므로 데이터를 증폭시키지 않는다

#D:\_data\Image\brain\train

xy_train = train_datagen.flow_from_directory('../_data/Image/brain/train', target_size=(150, 150), batch_size=200, class_mode='binary', shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/brain/test', target_size=(150, 150), batch_size=200, class_mode='binary')
# Found 120 images belonging to 2 classes.

print(xy_train[0][0].shape, xy_train[0][1].shape)   # (160, 150, 150, 3) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape)   # (120, 150, 150, 3) (120,)

np.save('./_save_npy/keras47_5_train_x.npy',arr=xy_train[0][0])
np.save('./_save_npy/keras47_5_train_y.npy',arr=xy_train[0][1])
np.save('./_save_npy/keras47_5_test_x.npy',arr=xy_test[0][0])
np.save('./_save_npy/keras47_5_test_y.npy',arr=xy_test[0][1])


