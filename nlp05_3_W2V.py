from unittest import skip
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sympy import Mod

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
# print(len(documents))  # 11314
# print(documents[1])

import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
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

# 단어 토큰화
def tokenize(d):
    return word_tokenize(d)


import pandas as pd

news_df = pd.DataFrame({'article':documents})
# print(len(news_df))  # 11314

news_df.replace("", float("NaN"), inplace=True)
news_df.dropna(inplace=True)
# print(len(news_df))  # 11096

news_df['article'] = news_df['article'].apply(clean_text)
# print(news_df['article'])

news_df['article'] = news_df['article'].apply(clean_stopword)
# print(news_df['article'])

tokenized_news = news_df['article'].apply(tokenize)
tokenized_news = tokenized_news.to_list()

import numpy as np

drop_news = [index for index, sentence in enumerate(tokenized_news) if len(sentence) <= 1]
news_texts = np.delete(tokenized_news, drop_news, axis=0)
# print(len(news_texts))  # 10945

from tensorflow.keras.preprocessing.text import Tokenizer

news_2000 = news_texts[:2000]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_2000)

idx2word = {value:key for key, value in tokenizer.word_index.items()}  # value:key 으로 inverse 시킴
sequences = tokenizer.texts_to_sequences(news_2000) 

vocab_size = len(tokenizer.word_index) + 1
# print(vocab_size)  # 29769
# print(sequences[1])
# [1263, 457, 2, 60, 119, 419, 61, 1374, 22, 69, 3498, 397, 6874, 412, 1173, 373, 2256, 458, 59, 12478, 458, 1900, 
# 3850, 397, 22, 10, 4325, 8749, 177, 303, 136, 154, 664, 12479, 316, 12480, 15, 12481, 4, 790, 12482, 12483, 4917, 8750]




########################################### 1. skipgrams  ##################################################

from tensorflow.keras.preprocessing.sequence import skipgrams

skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in sequences[:10]]

pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("{:s}({:d}), {:s}({:d}) -> {:d}".format(
        idx2word[pairs[i][0]], pairs[i][0],
        idx2word[pairs[i][1]], pairs[i][1],
        labels[i]))

# print(len(skip_grams))  # 10
# print(len(pairs))   # 2420
# print(len(labels))  # 2420

skip_grams = [skipgrams(seq, vocabulary_size=vocab_size, window_size=10) for seq in sequences]


# skipgrams 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input, Dot
from tensorflow.keras.utils import plot_model

embed_size = 50

def word2vec():
    target_inputs = Input(shape=(1, ), dtype='int32')
    target_embedding = Embedding(vocab_size, embed_size)(target_inputs)
    
    context_inputs = Input(shape=(1, ), dtype='int32')
    context_embedding = Embedding(vocab_size, embed_size)(context_inputs)
    
    dot_product = Dot(axes=2)([target_embedding, context_embedding])
    dot_product = Reshape((1, ), input_shape=(1, ))(dot_product)
    output = Activation('sigmoid')(dot_product)
    
    model = Model(inputs=[target_inputs, context_inputs], outputs = output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model

model = word2vec()
model.summary()
print(plot_model(model, show_shapes=True, show_layer_names=True))


for epoch in range(1, 11):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X, Y)
        
    print('Epoch:', epoch, 'Loss:', loss)
    

import gensim

f = open('skipgram.txt', 'w')
f.write('{} {}\n'.format(vocab_size-1, embed_size))
vectors = model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()

skipgram = gensim.models.KeyedVectors.load_word2vec_format('skipgram.txt', binary=False)

skipgram.most_similar(positive=['soldier'])

skipgram.most_similar(positive=['world'])


        
