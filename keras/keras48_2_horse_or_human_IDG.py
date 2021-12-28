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

train_generator = train_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=10, class_mode='binary', subset='training')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
validation_generator = train_datagen.flow_from_directory('../_data/Image/horse-or-human', target_size=(50, 50), batch_size=10, class_mode='binary', subset='validation')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 719 images belonging to 2 classes.
# Found 308 images belonging to 2 classes.

# print(train_generator[0][0])
# print(train_generator[0][1])
# print(validation_generator[0][0])
# print(validation_generator[0][1])

# np.save('../save/_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# np.save('../save/_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# np.save('../save/_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# np.save('../save/_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])

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

# batch = 10
# alldata = len(train_generator)
# spe = alldata/batch

# print(alldata)=72 -> 이미 len(train_generator)는 train_generator를 batch_size로 나눈 길이임. 따라서 steps_per_epoch에 len(train_generator)를 그대로 넣어주면 됨.

start = time.time()
# model.fit(xy_train[0][0], xy_train[0][1]) steps_per_epoch= spe,
# hist = model.fit_generator(xy_datagen.flow(xy_train, xy_validation, batch_size=50), epochs=100, steps_per_epoch=spe, validation_steps=4, callbacks=[es])  # steps_per_epoch -> 전체데이터/batchsize = 160/5=32
hist = model.fit_generator(train_generator, epochs=100, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=4, callbacks=[es])  # steps_per_epoch -> 전체데이터/batchsize = 160/5=32
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')
# model.save('../save/_save/keras48_2_save_weights.h5')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# '''
# loss :  0.623421311378479
# val_loss :  0.675041139125824
# acc :  0.680111289024353
# val_acc :  0.625
# '''

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
scores = model.evaluate_generator(validation_generator, steps=5)
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
validation_generator.reset()
print(validation_generator.class_indices)
# {'cats': 0, 'dogs': 1}
if(classes[0][0]<=0.5):
    horse = 100 - classes[0][0]*100
    print(f"당신은 {round(horse,2)} % 확률로 말 입니다")
elif(classes[0][0]>=0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 사람 입니다")
else:
    print("ERROR")

'''
-- Predict --
[[0.5461054]]
{'horses': 0, 'humans': 1}
당신은 54.61 % 확률로 사람 입니다
'''

# import numpy as np
# import pandas as pd

# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt

# pic_path = '../_data/Image/aram'
# model_path = '../save/_save/keras48_2_save_weights.h5'

# def load_my_image(img_path,show=False):
#     img = image.load_img(img_path, target_size=(50,50))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis = 0)
#     img_tensor /=255.
    
#     if show:
#         plt.imshow(img_tensor[0])    
#         plt.append('off')
#         plt.show()
    
#     return img_tensor

# if __name__ == '__main__':
#     model = load_model(model_path)
#     new_img = load_my_image(pic_path)
#     pred = model.predict(new_img)
#     horse = pred[0][0]*100
#     human = pred[0][1]*100
#     if horse > human:
#         print(f"당신은 {round(horse,2)} % 확률로 horse 입니다")
#     else:
#         print(f"당신은 {round(human,2)} % 확률로 human 입니다")



