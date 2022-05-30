# 어간 추출(Stemming)과 표제어 추출(Lemmatization)
# 택스트 전처리의 목적은 말뭉치(Corpus)로부터 복잡성을 줄이는 것 
# 어간 추출과 표제어 추출 역시 말뭉치의 복잡성을 줄여주는 텍스트 정규화 기법
# 어간 추출(Stemming)과 표제어 추출(Lemmatization)은 단어의 원형을 찾는 것

from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked')) # work work work
print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused')) # amus amus amus
print(stemmer.stem('happier'), stemmer.stem('happiest')) # happy happiest
print(stemmer.stem('fancier'), stemmer.stem('fanciest')) # fant fanciest
print(stemmer.stem('was'), stemmer.stem('love')) # was lov


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked'))  # work work work
print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused')) # amus amus amus
print(stemmer.stem('happier'), stemmer.stem('happiest')) # happier happiest
print(stemmer.stem('fancier'), stemmer.stem('fanciest')) # fancier fanciest
print(stemmer.stem('was'), stemmer.stem('love')) # wa love


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing'), lemma.lemmatize('amuses'), lemma.lemmatize('amused')) # amusing amuses amused
print(lemma.lemmatize('happier'), lemma.lemmatize('happiest')) # happier happiest
print(lemma.lemmatize('fancier'), lemma.lemmatize('fanciest')) # fancier fanciest
print(lemma.lemmatize('was'), lemma.lemmatize('love')) # wa love

# 표제어 추출(Lemmatization)은 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있음
print(lemma.lemmatize('amusing','v'),lemma.lemmatize('amuses','v'),lemma.lemmatize('amused','v')) # amuse amuse amuse
print(lemma.lemmatize('happier','a'),lemma.lemmatize('happiest','a')) # happy happy
print(lemma.lemmatize('fancier','a'),lemma.lemmatize('fanciest','a')) # fancy fancy
print(lemma.lemmatize('was', 'v'), lemma.lemmatize('love', 'v'))  # be love




