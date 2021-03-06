# 함수형으로 구현하기

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터 
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '예람이가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고예요': 5, '만든': 6, '영화예요': 7,
#  '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15,
#  '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21,
#  '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '예람이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
# print(x)  # [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

# 각 문장들의 길이가 다 다르므로 모델링을 할 수 없다. shape이 맞아야 모델링을 할 수 있으므로 가장 긴 문장을 기준으로 shape을 맞춰준다. 빈 부분은 0으로 채워주는데 통상적으로 앞쪽에 채운다

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)  # maxlen은 맞출(가장 긴)문장길이, padding='post'는 뒤쪽에 0을 채우겠다. padding='pre'는 앞쪽에 0을 채우겠다.
print(pad_x)
print(pad_x.shape)   # (13, 5)  -> 텐서플로우안에서 데이터셋을 반환하면 넘파이로 바뀜 그래서 shape이 찍힘

word_size = len(token.word_index)
print('word_size : ', word_size)     # word_size :  27
print(np.unique(pad_x))   # [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 ]

# 원핫인코딩하면 뭘로 바껴? (13, 5) -> (13, 5, 28)
# 옥스포드 사전은? (13, 5, 1000000) -> 65만개??  이렇게하면 망함!!

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input   # layers에서 Embedding 좌표로 바꿔주겠다

#2. 모델구성

input1 = Input(shape=(5,))
dense1 = Dense(32)(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(16)(dense2)
dense4 = Dense(8, activation='relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# #                                                인풋은 (13, 5)
# #                   단어사전의 갯수               단어수, 길이
# # model.add(Embedding(input_dim=28, output_dim=10, input_length=5))  # Embedding 원핫인코딩 하지 않은 데이터를 벡터화시키는것
# # model.add(Embedding(28, 10, input_length=5))  
# model.add(Embedding(28, 10))  
# model.add(LSTM(32, activation='relu'))
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)

####################### 실습 : 소스 완성시키기(결과는 긍정인지, 부정인지) ######################
x_predict = '나는 반장이 정말 재미없다 정말'
# print(type(x_predict))
docs_x_predict = [x_predict]
# print(docs_x_predict)   # ['나는 반장이 정말 재미없다 정말']

x_predict2 = token.texts_to_sequences(docs_x_predict)

pad_x2 = pad_sequences(x_predict2, padding='pre', maxlen=5)  # maxlen은 맞출(가장 긴)문장길이, padding='post'는 뒤쪽에 0을 채우겠다. padding='pre'는 앞쪽에 0을 채우겠다.
# print(pad_x2)   # [[ 0  0  0  0 23]]
# print(pad_x2.shape)   # (1, 5)

final_predict = model.predict(pad_x2)

print(final_predict)

'''
acc :  0.5384615659713745
[[8.022137]]
'''