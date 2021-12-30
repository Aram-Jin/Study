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
                                   zoom_range=0.1, 
                                #    shear_range=0.7, 
                                   fill_mode='nearest')   # 1./255 minmax scale 지정 0~1사이로(사진에선 최대값이 255이므로), horizontal_flip은 상하반전, vertical_flip은 좌우반전, width_shift_range는 이미지 이동해도 인식(이미지증폭)

test_datagen = ImageDataGenerator(rescale=1./255)   #test데이터는 평가만 할 것이므로 데이터를 증폭시키지 않는다

xy_train = train_datagen.flow_from_directory('../_data/Image/cat_dog/training_set', target_size=(100, 100), batch_size=9000, class_mode='binary', shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 8005 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/cat_dog/test_set', target_size=(100, 100), batch_size=9000, class_mode='binary')
# Found 2023 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape)  # (8005, 100, 100, 3) (8005,)
# print(x_test.shape, y_test.shape)  # (2023, 100, 100, 3) (2023,)

augment_size = 1995   
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # randint : 랜덤한 정수값을 생성하겠다. 
# print(x_train.shape[0])   # 8005
# print(randidx)   # [7644  639 3740 ... 3304 1243 3083]
# print(np.min(randidx), np.max(randidx))   # 1 8001
# print(randidx.shape)   # (1995,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)   # (1995, 100, 100, 3)
# print(y_augmented.shape)   # (1995,)

x_train = np.concatenate((x_train, x_augmented))    # concatenate를 사용할때는 (())괄호 두개로 사용해주어야함
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)   # (10000, 100, 100, 3) (10000,)
# print(x_test.shape, y_test.shape)   # (2023, 100, 100, 3) (2023,)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(100, 100, 3))) 
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

start = time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=100, validation_split=0.2, callbacks=[es])  
end = time.time() - start
 
print("걸린시간 : ", round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('accuracy:', loss[1])


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
