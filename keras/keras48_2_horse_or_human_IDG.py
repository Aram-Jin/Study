import numpy as np
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
xy_datagen = ImageDataGenerator(rescale=1./255) 
xy = xy_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=500, class_mode='binary', shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 1027 images belonging to 2 classes.

# np.save('./_save_npy/keras48_2_x.npy',arr=xy[0][0])
# np.save('./_save_npy/keras48_2_y.npy',arr=xy[0][1])

x = np.load('./_save_npy/keras48_2_x.npy')
y = np.load('./_save_npy/keras48_2_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=66)

x_train, y_train = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, 
                                      height_shift_range=0.1, rotation_range=5, zoom_range=1.2, shear_range=0.7, fill_mode='nearest') 

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(50, 50, 3))) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32, validation_data=xy_test, validation_steps=4, callbacks=[es])  # steps_per_epochs -> 전체데이터/batchsize = 160/5=32

