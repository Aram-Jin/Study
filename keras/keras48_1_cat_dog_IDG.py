#http://www.kaggle.com/c/dogs-vs-cats/data
import numpy as np, time
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

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

xy_train = train_datagen.flow_from_directory('../_data/Image/cat_dog/training_set', target_size=(50, 50), batch_size=100, class_mode='binary', shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 8005 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/cat_dog/test_set', target_size=(50, 50), batch_size=100, class_mode='binary')
# Found 2023 images belonging to 2 classes.


# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000028AED255940>
# print(xy_train[31])       # 마지막배치
# print(xy_train[0][0])     # 첫번쨰 괄호 '0'은 첫번째배치 두번째괄호의 '0'은 첫번쨰 배치의 x값
# print(xy_train[0][1])
# print(xy_train[0][2])   #IndexError: tuple index out of range -> tuple은 수정이 안됨 list는 수정가능
# print(xy_train[0][0].shape, xy_train[0][1].shape)     # (100, 50, 50, 3) (100,)

# print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))   # <class 'tuple'>
# print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(50, 50, 3))) 
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

start = time.time()
hist = model.fit_generator(xy_train, epochs=1000, steps_per_epoch=161, validation_data=xy_test, validation_steps=4, callbacks=[es])  # steps_per_epochs -> 전체데이터/batchsize = 160/5=32
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
hist = model.evaluate_generator(xy_test, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

'''
loss :  1.182776927947998
val_loss :  0.6917902231216431
acc :  0.5018113851547241
val_acc :  0.48500001430511475
'''

# 샘플 케이스 경로지정
sample_directory = '../_data/Image/aram/'
sample_image = sample_directory + "aram.jpg"

# 샘플 케이스 확인
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

image_ = plt.imread(str(sample_image))
plt.title("Test Case")
plt.imshow(image_)
plt.axis('Off')
plt.show()

print("-- Evaluate --")
scores = model.evaluate_generator(xy_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
xy_test.reset()
print(xy_test.class_indices)
# {'cats': 0, 'dogs': 1}
if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
elif(classes[0][0]>=0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
else:
    print("ERROR")


'''    
-- Predict --
[[0.02125992]]
{'cats': 0, 'dogs': 1}
당신은 97.87 % 확률로 고양이 입니다
'''
