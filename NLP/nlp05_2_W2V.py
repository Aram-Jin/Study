import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import os
from tqdm import tqdm
import platform
# print(platform.architecture())  # ('64bit', 'WindowsPE')

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')
print(train_data[:5]) # 상위 5개 출력

print(len(train_data)) # 200000 : 리뷰 개수 출력 

##### 1) 데이터 전처리

# NULL 값 존재 유무
print(train_data.isnull().values.any())  # True

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # False : Null 값이 존재하는지 확인

print(len(train_data)) # 199992 : 리뷰 개수 출력

# 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(train_data[:5]) # 상위 5개 출력

# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)
    
    
# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))  # 리뷰의 최대 길이 : 72
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))  # 리뷰의 평균 길이 : 10.716703668146726
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()    


##### 2) Word2Vec으로 토큰화 된 네이버 영화 리뷰 데이터를 학습
from gensim.models import Word2Vec

model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

# 완성된 임베딩 매트릭스의 크기 확인
print(model.wv.vectors.shape)

##### 3) 결과 확인
print(model.wv.most_similar("최민식"))
# [('한석규', 0.8577019572257996), ('안성기', 0.854630172252655), ('이정재', 0.8413209319114685), ('김명민', 0.8277357816696167), ('설경구', 0.8263182044029236), 
#  ('최민수', 0.8256040215492249), ('주진모', 0.8159744739532471), ('크리스찬', 0.8156553506851196), ('한예리', 0.8107945919036865), ('채민서', 0.8091418147087097)]

print(model.wv.most_similar("히어로"))
# [('슬래셔', 0.8586012125015259), ('느와르', 0.8425800204277039), ('물', 0.8388407826423645), ('호러', 0.83847004175186163)]