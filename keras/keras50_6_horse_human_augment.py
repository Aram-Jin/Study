import numpy as np, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
                                #    vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                #    rotation_range=5, 
                                   zoom_range=1.2, 
                                #    shear_range=0.7, 
                                   fill_mode='nearest',
                                   validation_split=0.3) 

train_generator = train_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=10, class_mode='binary', subset='training')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
validation_generator = train_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=10, class_mode='binary', subset='validation')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌

test_datagen = ImageDataGenerator(rescale=1./255)