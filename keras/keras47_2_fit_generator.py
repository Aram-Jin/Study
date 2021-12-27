import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import sequential


#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
                                   vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   rotation_range=5, 
                                   zoom_range=1.2, 
                                   shear_range=0.7, 
                                   fill_mode='nearest')   # 1./255 minmax scale 지정 0~1사이로(사진에선 최대값이 255이므로), horizontal_flip은 상하반전, vertical_flip은 좌우반전, width_shift_range는 이미지 이동해도 인식(이미지증폭)

test_datagen = ImageDataGenerator(rescale=1./255)   #test데이터는 평가만 할 것이므로 데이터를 증폭시키지 않는다

#D:\_data\Image\brain\train

xy_train = train_datagen.flow_from_directory('../_data/Image/brain/train', target_size=(150, 150), batch_size=5, class_mode='binary', shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/brain/test', target_size=(150, 150), batch_size=5, class_mode='binary')
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002B85C344F70>
# print(xy_train[31])       # 마지막배치
# print(xy_train[0][0])     # 첫번쨰 괄호 '0'은 첫번째배치 두번째괄호의 '0'은 첫번쨰 배치의 x값
# print(xy_train[0][1])
# print(xy_train[0][2])   #IndexError: tuple index out of range -> tuple은 수정이 안됨 list는 수정가능
print(xy_train[0][0].shape, xy_train[0][1].shape)     # (5, 150, 150, 3) (5,)

print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))   # <class 'tuple'>
print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (150,150,3))) 
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

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32, validation_data=xy_test, validation_steps=4, callbacks=[es])  # steps_per_epochs -> 전체데이터/batchsize = 160/5=32


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 점심때 그래프 그리기~!!!

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

'''
loss :  0.6786040663719177
val_loss :  0.7193442583084106
acc :  0.643750011920929
val_acc :  0.6000000238418579
'''
'''
loss :  0.6893249750137329
val_loss :  0.6901830434799194
acc :  0.5562499761581421
val_acc :  0.6000000238418579
'''



import matplotlib.pyplot as plt
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['acc'], marker='.', c='black', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='green', label='val_acc')
plt.xlabel('epoch')
plt.grid()
plt.legend(loc='upper right')
plt.show()


# # summarize history for accuracy
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()



# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()
