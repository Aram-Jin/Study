import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display
import nltk
import tensorflow as tf
from tensorflow import keras

import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

root_dir = "../data/news"
nltk.download('stopwords')
nltk.download("punkt")

train = pd.read_csv(os.path.join(root_dir, "train.csv"), index_col="id")
test = pd.read_csv(os.path.join(root_dir, "test.csv"), index_col="id")
#df_submission = pd.read_csv(os.path.join(root_dir, "sample_submission.csv"))
submission = pd.read_csv("../data/news/sample_submission.csv")

# train["length"] = train.text.map(len)
# test["length"] = test.text.map(len)
# print(display(train.head(5)))
# print(display(test.head(5)))

import re 

def clean_text(texts): 
  corpus = [] 
  for i in range(0, len(texts)): 

    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>]', '',texts[i]) #@%*=()/+ 와 같은 문장부호 제거
    review = re.sub(r'\d+','', review)#숫자 제거
    review = review.lower() #소문자 변환
    review = re.sub(r'\s+', ' ', review) #extra space 제거
    review = re.sub(r'<[^>]+>','',review) #Html tags 제거
    review = re.sub(r'\s+', ' ', review) #spaces 제거
    review = re.sub(r"^\s+", '', review) #space from start 제거
    review = re.sub(r'\s+$', '', review) #space from the end 제거
    review = re.sub(r'_', ' ', review) #space from the end 제거
    corpus.append(review) 
  
  return corpus

temp = clean_text(train['text']) #메소드 적용
train['text'] = temp

temp = clean_text(test['text'])
test['text'] = temp

# print(train.head())
# print(test.head())

# x = train.text
# y = train.target

val_count = train['target'].value_counts() # 유니크값의 개수를 확인합니다.

for i in range(0,20):
  print(f'라벨 {i}인 리뷰 개수:', val_count[i])

plt.style.use("ggplot")

# 히스토그램 을 사용해서 데이터의 분포를 살펴봅니다.
feature = train['target']

plt.figure(figsize=(10,7.5)) # 그래프 이미지 크기 설정
plt.suptitle("Bar Plot", fontsize=30) # 부제목과 폰트 크기 설정

plt.title('label', fontsize=20) # 제목과 폰트 크기 설정
temp = feature.value_counts() # feature 변수의 변수별 개수 계산
plt.bar(temp.keys(), temp.values, width=0.5, color='b', alpha=0.5) # 막대그래프 생성
plt.xticks(temp.keys(), fontsize=12) # x축 값, 폰트 크기 설정
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 레이아웃 설정
# plt.show() # 그래프 나타내기

str_len_mean = np.mean(train['text'].str.len()) # 리뷰 길이의 평균값 계산
print('뉴스의 평균 길이 :',round(str_len_mean,0))   # 뉴스의 평균 길이 : 1020.0

# 데이터 필터링을 위한 마스크 설정
for i in range(0, 20):
  globals()['mask_{}'.format(i)] = (train.target == i)

# 전체 및 그룹 집합을 설정합니다.
df_train = train.text.copy() # 전체 train 데이터

for i in range(0, 20):
  globals()['df_{}'.format(i)] = train.loc[globals()['mask_{}'.format(i)],:].text # 20가지 라벨 각각에 해당하는 데이터를 df0~19로 할당
# #스무가지로 나뉜 집합을 리스트로 묶어줍니다.
compare = [df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19]

# 히스토그램 을 사용해서 데이터의 분포를 살펴봅니다.
plt.figure(figsize=(40,20))
plt.suptitle("Histogram: news length", fontsize=40)
name = ['alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'misc.forsale',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space',
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc'] # 제목으로 사용할 문자열 (라벨의 실제 이름)

for i in range(len(compare)):
    text = compare[i]
    string_len = [len(x) for x in text]    
    plt.subplot(5,4,i+1) # 행 개수/ 열 개수/ 해당 그래프 표시 순서
    plt.title(name[i], fontsize=20)
    plt.axis([0, 50000, 0, 10])  #x축 시작, 끝 / y축 시작, 끝
    plt.hist(string_len, alpha=0.5, color='orange') # 히스토그램 생성, alpha: 그래프의 투명도 조절
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import nltk
import pickle
from nltk.corpus import stopwords
from os import path
nltk.download('all')

def df2str(df):
    
    s = [s for s in df]
    document = ""
    for i in range(len(s)):
        document += s[i]
    return document

word_tokens = nltk.word_tokenize(df2str(df_0))

# pos_tag()의 입력값으로는 단어의 리스트가 들어가야 한다.
tokens_pos = nltk.pos_tag(word_tokens)

# 명사는 NN을 포함하고 있음을 알 수 있음
NN_words = []
for word, pos in tokens_pos:
    if 'NN' in pos:
        NN_words.append(word)
        
# 명사의 경우 보통 복수 -> 단수 형태로 변형
wlem = nltk.WordNetLemmatizer()
lemmatized_words = []
for word in NN_words:
    new_word = wlem.lemmatize(word)
    lemmatized_words.append(new_word)
    
stopwords_list = stopwords.words('english') #nltk에서 제공하는 불용어사전 이용
#print('stopwords: ', stopwords_list)
unique_NN_words = set(lemmatized_words)
final_NN_words = lemmatized_words

# 불용어 제거
for word in unique_NN_words:
    if word in stopwords_list:
        while word in final_NN_words: final_NN_words.remove(word)
        
from collections import Counter
c = Counter(final_NN_words)
k = 10

print(c.most_common(k)) # 빈도수 기준 상위 k개 단어 출력

top_10 = c.most_common(k) # 추출한 명사 중 상위 10개
keys = [top_10[i][0] for i in range(len(top_10))]
values = [top_10[i][1] for i in range(len(top_10))]

plt.figure(figsize=(10,7.5))
plt.suptitle("Bar Plot", fontsize=30)
plt.title('total reviews', fontsize=20)
plt.bar(keys, values, width=0.5, color='b', alpha=0.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

noun_text = ''
for word in final_NN_words:
    noun_text = noun_text +' '+word

wordcloud = WordCloud(max_font_size=50, #가장 큰 폰트 크기 제한
                      width=500, #너비
                      height=300, #높이
                      background_color='white', #배경 색상
                      relative_scaling=.2 #상대적인 크기
                      ).generate(noun_text)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

# [('god', 190), ('people', 176), ('argument', 122), ('time', 105), ('thing', 99), ('way', 97), ('book', 90), 
#  ('question', 89), ('something', 87), ('religion', 84)]

# print(train)

# from sklearn.feature_extraction.text import CountVectorizer #sklearn 패키지의 CountVectorizer import
# sample_vectorizer = CountVectorizer() 

import pandas as pd
import numpy as np
import os
from glob import glob

x = train.text
y = train.target

# print(np.unique(y, return_counts=True))
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (9233, 20)

pd_y = pd.DataFrame(y)
print(pd_y)

y = pd_y.apply(lambda x: x.argmax(), axis=1).values
print(y.shape)  # (9233,)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf = True, ngram_range = (1, 2))
vectorizer.fit(np.array(x))

sorted(vectorizer.vocabulary_.items())
print(vectorizer.vocabulary_)

x_vec = vectorizer.transform(x)
result_vec = vectorizer.transform(test["text"])

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

x_train, x_test, y_train, y_test = train_test_split(x_vec, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

model = XGBClassifier() 
model.fit(x_train, y_train, verbose=1, eval_set = [(x_test, y_test)], eval_metric= 'merror', early_stopping_rounds=500)

score = model.score(x_test, y_test)
print("score: ", round(score, 4))

pred = model.predict(result_vec)
submission["target"] = pred
print(submission)

submission.to_csv("Y_SMOTE_submission.csv", index = False)
