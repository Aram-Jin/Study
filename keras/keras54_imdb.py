from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# print(x_train.shape, x_test.shape)   # (25000,) (25000,)
# print(y_train.shape, y_test.shape)   # (25000,) (25000,)
# print(np.unique(y_train))   # [0 1]

# print(x_train[0], y_train[0])  
# print(len(x_train[0]), len(x_train[1]))   # 218 189

# print(type(x_train), type(y_train))      # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(type(x_test), type(y_test))       # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# print("최대길이 : ", max(len(i) for i in x_train))   # 최대길이 :  2494
# print("평균길이 : ", sum(map(len, x_train)) / len(x_train))   # 평균길이 :  238.71364

x_train = pad_sequences(x_train, padding='pre', maxlen=200, truncating='pre')
# print(x_train.shape)    # (25000, 200)

x_test = pad_sequences(x_test, padding='pre', maxlen=200, truncating='pre')
# print(x_test.shape)    # (25000, 200)

# print(x_train.shape, y_train.shape)    # (25000, 200) (25000, )
# print(x_test.shape, y_test.shape)    # (25000, 200) (25000, )

# 2. 모델구성
model = Sequential()
#                   단어사전의 갯수                 단어수, 길이
model.add(Embedding(input_dim=10000, output_dim=68, input_length=200))  # Embedding 원핫인코딩 하지 않은 데이터를 벡터화시키는것
model.add(LSTM(48, activation='relu'))
model.add(Dense(36))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))
model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=100, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)
print('acc : ', acc)
'''
acc :  [nan, 0.5]
'''

#########################################################################################

# word_to_index = reuters.get_word_index()
# # print(word_to_index)
# # print(sorted(word_to_index.items()))  
# import operator
# print(sorted(word_to_index.items(), key = operator.itemgetter(1)))

# index_to_word = {}
# for key, value in word_to_index.items():
#     index_to_word[value+3] = key
    
# for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
#     index_to_word[index] = token
    
# print(' '.join([index_to_word[index] for index in x_train[0]]))


  