from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

# print(x_train[0])

print(x_train, len(x_train), len(x_test))     # 8982, 2246
print(y_train[0])    # 3
print(np.unique(y_train))   # 46개의 뉴스 카테고리
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]  -> 46개

print(type(x_train), type(y_train))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape)   # (8982,) (8982,)

print(len(x_train[0]), len(x_train[1]))   # 87 56
print(type(x_train[0]), type(x_train[1]))   # <class 'list'> <class 'list'>

# print("뉴스기사의 최대길이 : ", max(len(x_train))) # TypeError: 'int' object is not iterable
print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))   # 뉴스기사의 최대길이 :  2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))   # 뉴스기사의 평균길이 :  145.5398574927633
# map(len, x_train) -> x_train에 대한 len만큼을 반환해준다.  sum은 합

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
print(x_train.shape)    # (8982, 2376)  -> (8982, 100)

x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
print(x_test.shape)    # (2246, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)    # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)    # (2246, 100) (2246, 46)

print(type(x_train), type(y_train))    
print(type(x_test), type(y_test)) 
'''
<class 'numpy.ndarray'> <class 'numpy.ndarray'>
<class 'numpy.ndarray'> <class 'numpy.ndarray'>
'''

# #2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout  

# model = Sequential()
# #                                                인풋은 (13, 5)
# #                   단어사전의 갯수               단어수, 길이
# model.add(Embedding(input_dim=10000, output_dim=68, input_length=100))  # Embedding 원핫인코딩 하지 않은 데이터를 벡터화시키는것
# # model.add(Embedding(28, 10, input_length=5))  
# # model.add(Embedding(28, 10))  
# model.add(LSTM(48, activation='relu'))
# model.add(Dense(36))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(46, activation='softmax'))
# model.summary()


# #3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

# model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])

# #4. 평가, 예측
# acc = model.evaluate(x_test, y_test)[1]
# print('acc : ', acc)


#########################################################################################

word_to_index = reuters.get_word_index()
# print(word_to_index)
# print(sorted(word_to_index.items()))  
import operator
print(sorted(word_to_index.items(), key = operator.itemgetter(1)))

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key
    
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token
    
print(' '.join([index_to_word[index] for index in x_train[0]]))

