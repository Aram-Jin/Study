import numpy as np, time
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

# print(x_train.shape, y_train.shape)   # (719, 50, 50, 3) (719,)
# print(x_test.shape, y_test.shape)   # (308, 50, 50, 3) (308,)

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

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, callbacks=[es])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')
# model.save_weights('./_save/keras48_1_save_weights.h5')

#4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test)
print('loss : ', loss_acc[0])
print('accuracy : ', loss_acc[1])

'''
걸린시간 :  28.506 초
loss :  1.9018152952194214
accuracy :  0.6655844449996948
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
scores = model.evaluate_generator(x_test, y_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
y_test.reset()
print(y_test.class_indices)
# {'cats': 0, 'dogs': 1}
if(classes[0][0]<=0.5):
    horse = 100 - classes[0][0]*100
    print(f"당신은 {round(horse,2)} % 확률로 말 입니다")
elif(classes[0][0]>=0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 사람 입니다")
else:
    print("ERROR")


