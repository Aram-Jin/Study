from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])   # 리스트형식 -> 여러문장을 사용하기 위해서

print(token.word_index)   # {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}  -> 자주나오는게 먼저, 그다음은 앞에있는 순서대로 인덱싱시켜줌

x = token.texts_to_sequences([text])
print(x)    # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]  -> 텍스트 형식을 수치화 했을 경우 숫자가 클수록 가치가 있다고 판단할 수 있는 오류가 있으므로 원핫인코딩을 꼭해주어야함

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print('word_size: ', word_size)  # 7 -> 토큰(단어)에는 7개의 종류가 있다

x = to_categorical(x)
print(x)
print(x.shape)  
# [[[0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1.]]] 
# (1, 9, 8)     -> '1' : 위에서 [text]로 정의해주어서 제일 바깥에 []가 한개 더 생성됨, '9': 9개의 어절(띄어쓰기), '8' : 0~7(to_categorical 해주어서 0들어감)

