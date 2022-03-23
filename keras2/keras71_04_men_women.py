import numpy as np, time
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
                                   vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   rotation_range=5, 
                                   zoom_range=1.2, 
                                   shear_range=0.7, 
                                   fill_mode='nearest',
                                   validation_split=0.3) 

train_generator = train_datagen.flow_from_directory('../_data/Image/men_women', target_size=(50, 50), batch_size=10, class_mode='binary', subset='training')   
validation_generator = train_datagen.flow_from_directory('../_data/Image/men_women', target_size=(50, 50), batch_size=10, class_mode='binary', subset='validation')  

np.save('./_save_npy/keras48_1_train_x.npy',arr=train_generator[0][0])
np.save('./_save_npy/keras48_1_train_y.npy',arr=train_generator[0][1])
np.save('./_save_npy/keras48_1_test_x.npy',arr=xy_test[0][0])
np.save('./_save_npy/keras48_1_test_y.npy',arr=xy_test[0][1])

# x_train = np.load('../save/_save_npy/keras48_1_train_x.npy')
# y_train = np.load('../save/_save_npy/keras48_1_train_y.npy')
# x_test = np.load('../save/_save_npy/keras48_1_test_x.npy')
# y_test = np.load('../save/_save_npy/keras48_1_test_y.npy')

# print(x_train)
# print(x_train.shape)   # (50, 50, 50, 3)