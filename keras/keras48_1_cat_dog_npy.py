#http://www.kaggle.com/c/dogs-vs-cats/data
import numpy as np, time
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
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

xy_train = train_datagen.flow_from_directory('../_data/Image/cat_dog/training_set', target_size=(50, 50), batch_size=50, class_mode='binary', shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 8005 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/cat_dog/test_set', target_size=(50, 50), batch_size=50, class_mode='binary')
# Found 2023 images belonging to 2 classes.
# print(xy_train[0][0].shape, xy_train[0][1].shape)   # (50, 50, 50, 3) (50,)
# print(xy_test[0][0].shape, xy_test[0][1].shape)   # (50, 50, 50, 3) (50,)

# np.save('./_save_npy/keras48_1_train_x.npy',arr=xy_train[0][0])
# np.save('./_save_npy/keras48_1_train_y.npy',arr=xy_train[0][1])
# np.save('./_save_npy/keras48_1_test_x.npy',arr=xy_test[0][0])
# np.save('./_save_npy/keras48_1_test_y.npy',arr=xy_test[0][1])

x_train = np.load('../save/_save_npy/keras48_1_train_x.npy')
y_train = np.load('../save/_save_npy/keras48_1_train_y.npy')
x_test = np.load('../save/_save_npy/keras48_1_test_x.npy')
y_test = np.load('../save/_save_npy/keras48_1_test_y.npy')

# print(x_train)
# print(x_train.shape)   # (50, 50, 50, 3)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (50,50,3))) 
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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.2, callbacks=[es])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test)
print('loss : ', loss_acc[0])
print('accuracy : ', loss_acc[1])

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

'''
loss :  1.290297031402588
accuracy :  0.5199999809265137
loss :  0.07104019820690155
val_loss :  1.4927408695220947
acc :  0.9750000238418579
val_acc :  0.6000000238418579
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
classes = model.predict(images, batch_size=50)
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
[[0.0093929]]
{'cats': 0, 'dogs': 1}
당신은 99.06 % 확률로 고양이 입니다
'''

'''
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/Image/aram'
model_path = './_save/keras48_1_save_weights.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(50,50))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    dog = pred[0][0]*100
    cat = pred[0][1]*100
    if cat > dog:
        print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
   
   
load_my_image(pic_path,show=False)
'''

