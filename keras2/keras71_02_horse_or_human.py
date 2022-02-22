import numpy as np, time
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
# train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
#                                    horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
#                                    vertical_flip=True, 
#                                    width_shift_range=0.1, 
#                                    height_shift_range=0.1, 
#                                    rotation_range=5, 
#                                    zoom_range=1.2, 
#                                    shear_range=0.7, 
#                                    fill_mode='nearest',
#                                    validation_split=0.3) 

# train_generator = train_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=10, class_mode='binary', subset='training')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# validation_generator = train_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=10, class_mode='binary', subset='validation')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 719 images belonging to 2 classes.
# Found 308 images belonging to 2 classes.

# np.save('../save/_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# np.save('../save/_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# np.save('../save/_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# np.save('../save/_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])

x_train = np.load('../save/_save_npy/keras48_2_train_x.npy')
y_train = np.load('../save/_save_npy/keras48_2_train_y.npy')
x_test = np.load('../save/_save_npy/keras48_2_test_x.npy')
y_test = np.load('../save/_save_npy/keras48_2_test_y.npy')

print(x_train.shape, y_train.shape)   # (719, 50, 50, 3) (719,)
print(x_test.shape, y_test.shape)   # (308, 50, 50, 3) (308,)


#2.모델구성
denseNet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(50, 50, 3))

model = Sequential()
model.add(denseNet121)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', verbose=1, factor=0.5)  #-> 5번 만에 갱신이 안된다면 (factor=0.5) LR을 50%로 줄이겠다

start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# learning_rate :  0.0001
# loss :  0.6265
# accuracy :  0.8019
# 걸린시간 : 74.6872
