import numpy as np
from sklearn import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

xy_train = train_datagen.flow_from_directory('../_data/Image/brain/train', target_size=(150, 150), batch_size=5, class_mode='categorical', 
                                             shuffle='True', seed=66, color_mode='grayscale', save_to_dir='../_temp/')   #save_to_dir -> 건드린 데이터를 해당 폴더에 저장하여 출력해줌
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/brain/test', target_size=(150, 150), batch_size=5, class_mode='categorical')
# # Found 120 images belonging to 2 classes.

# # from sklearn.datasets import load_boston
# # datasets = load_boston()
# # # print(datasets)

print(xy_train)
# # # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002B85C344F70>
print(xy_train[31])       # 마지막배치
print(xy_train[0][0])     # 첫번쨰 괄호 '0'은 첫번째배치 두번째괄호의 '0'은 첫번쨰 배치의 x값
print(xy_train[0][1])
# # # print(xy_train[0][2])   #IndexError: tuple index out of range -> tuple은 수정이 안됨 list는 수정가능
print(xy_train[0][0].shape, xy_train[0][1].shape)     # (5, 150, 150, 1) (5, 2)   -> color_mode를 grayscale로 하면 

# # print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# # print(type(xy_train[0]))   # <class 'tuple'>
# # print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# # print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

'''
color_mode
save_to_dir
'''