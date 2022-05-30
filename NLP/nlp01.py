# 1. 텍스트 전처리

# (1) 문장 토큰화
from nltk import sent_tokenize
import nltk
nltk.download('punkt')

text_sample = 'The Mattrix is everywhere its all around us, here even in this room. You can see it out your window or on your television. You feel it when you go to work,or go to church or pay your taxes.'
sentences = sent_tokenize(text = text_sample)

print(type(sentences), len(sentences))  # <class 'list'> 3
print(sentences)  
# ['The Mattrix is everywhere its all around us, here even in this room.', 'You can see it out your window or on your television.', 
# 'You feel it when you go to work,or go to church or pay your taxes.']

# 각각의 문장으로 구성된 list 객체가 반환됨(반환된 list객체는 3개의 문장으로 된 문자열을 가지고 있음)


# (2) 단어 토큰화
from nltk import word_tokenize

sentence = "The Mattrix is everywhere its all around us, here even in this room"
words = word_tokenize(sentence)

print(type(words), len(words))  # <class 'list'> 14
print(words)  # ['The', 'Mattrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room']



# (3) 함수 정의
from nltk import word_tokenize, sent_tokenize

# 여러개의 문장으로 된 입력 데이터를 문장별로 단어 토큰화하게 만드는 함수 생성
def tokenize_text(text):
    
    # 문장별로 분리 토큰
    sentences = sent_tokenize(text)
    # 분리된 문장별 단어 토큰화
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

# 여러 문장에 대해 문장별 토큰화 수행
word_tokens = tokenize_text(text_sample)   # <class 'list'> 3
print(type(word_tokens), len(word_tokens))
print(word_tokens)  
# [['The', 'Mattrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.'], ['You', 'can', 'see', 'it', 'out', 'your', 
# window', 'or', 'on', 'your', 'television', '.'], ['You', 'feel', 'it', 'when', 'you', 'go', 'to', 'work', ',', 'or', 'go', 'to', 'church', 'or', 'pay', 'your', 'taxes', '.']]


##### 한계 : 단어별 하나씩 토큰화 하게 되면 문맥적 의미가 무시됨