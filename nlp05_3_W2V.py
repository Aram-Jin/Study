import pandas as pd
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
# print(len(documents))  # 11314
# print(documents[1])

import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopword')
nltk.download('punkt')

# 알파벳 이외의 문자 제거
def clean_text(d):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', d)
    return text

# 소문자로 바꾸고 길이가 3이하인 문자 제거
def clean_stopword(d):
    stop_words = stopwords.words('english')
    return ' '.join([w.lower() for w in d.split() if w not in stop_words and len(w) > 3])

def tokenize(d):
    return word_tokenize(d)


import pandas as pd

news_df = pd.DataFrame({'article':documents})
print(len(news_df))  # 11314

news_df.replace("", float("NaN"), inplace=True)
news_df.dropna(inplace=True)
print(len(news_df))  # 11096

news_df['article'] = news_df['article'].apply(clean_text)
# print(news_df['article'])

news_df['article'] = news_df['article'].apply(clean_stopword)
# print(news_df['article'])

tokenized_news = news_df['article'].apply(tokenize)
tokenized_news = tokenized_news.to_list()

import numpy as np

drop_news = [index for index, sentence in enumerate(tokenized_news) if len(sentence) <= 1]
news_texts = np.delete(tokenized_news, drop_news, axis=0)
print(len(news_texts))  # 10945

from tensorflow.keras.preprocessing.text import Tokenizer

news_2000 = news_texts[:2000]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_2000)

idx2word = {value:key for key, value in tokenizer.word_index.items()}
sequences = tokenizer.texts_to_sequences(news_2000)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

print(sequences[1])