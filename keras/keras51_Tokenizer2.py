from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'
text2 = '나는 매우 매우 질생긴 지구용사 태권브이.'

token = Tokenizer()

token.fit_on_texts([text1, text2])   # 리스트형식 -> 여러문장을 사용하기 위해서
print(token.word_index)   # {'매우': 1, '나는': 2, '진짜': 3, '마구': 4, '맛있는': 5, '밥을': 6, '먹었다': 7, '질생긴': 8, '지구용사': 9, '태권브이': 10} -> 자주나오는게 먼저, 그다음은 앞에있는 순서대로 인덱싱시켜줌

# Version1

text3 = text1 + text2

x = token.texts_to_sequences([text3])
print(x)    # [[2, 3, 1, 5, 6, 3, 4, 4, 7, 2, 1, 1, 8, 9, 10]]       # list는 shape을 제공하지 않으므로 len으로 봐야한다

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print('word_size: ', word_size)  # word_size:  10

x = to_categorical(x)
print(x)
print(x.shape)  
# [[[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 15, 11)


# # Version2

# x = token.texts_to_sequences([text1,text2])
# print(x)    # [[2, 3, 1, 5, 6, 3, 4, 4, 7], [2, 1, 1, 8, 9, 10]]

# x = x[0] + x[1]
# print(x)    # [2, 3, 1, 5, 6, 3, 4, 4, 7, 2, 1, 1, 8, 9, 10]

# from tensorflow.keras.utils import to_categorical
# word_size = len(token.word_index)
# print('word_size: ', word_size)  # word_size:  10

# x = to_categorical(x)
# print(x)
# print(x.shape)  

# # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# # (15, 11)