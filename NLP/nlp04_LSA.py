# LSA
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print(len(documents))  # 11314
print(dataset.target_names)
# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 
# 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
# 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


# 1. 전처리
news_df = pd.DataFrame({'document': documents})

# 알파벳 이외의 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
# 길이가 3이하인 문자 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
# 소문자로 바꾸기
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())


# 2. TF-IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # 1,000개의 단어만 추출
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])

print(X.shape) # (11314, 1000) DTM의 행렬 크기 반환


# 3. Truncated SVD를 활용하여 토픽 모델링
from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

print(svd_model.components_.shape)  # (20, 1000)  Vt 행렬의 크기
print(svd_model.singular_values_)
# [17.15952833  9.93882749  8.17139855  7.92032011  7.62377374  7.5257242
#   7.25096862  7.00623237  6.88289372  6.85602044  6.68476301  6.56045782
#   6.52895929  6.42222944  6.33939436  6.21686249  6.17477882  6.09487639
#   6.00247117  5.90654237]

terms = vectorizer.get_feature_names()
print(len(terms))  # 1000 : terms의 길이는 총 단어 피처 수인 1000이 됨

# 20개의 토픽에 대해 주요 단어를 나열
n = 8
components = svd_model.components_
for index, topic in enumerate(components):
    print('Topic %d: '%(index + 1), [terms[i] for i in topic.argsort()[: -n - 1: -1]])

'''
Topic 1:  ['just', 'like', 'know', 'people', 'think', 'does', 'good', 'time']
Topic 2:  ['thanks', 'windows', 'card', 'drive', 'mail', 'file', 'advance', 'files']
Topic 3:  ['game', 'team', 'year', 'games', 'drive', 'season', 'good', 'players']
Topic 4:  ['drive', 'scsi', 'disk', 'hard', 'problem', 'drives', 'just', 'card']
Topic 5:  ['drive', 'know', 'thanks', 'does', 'just', 'scsi', 'drives', 'hard']
Topic 6:  ['just', 'like', 'windows', 'know', 'does', 'window', 'file', 'think']
Topic 7:  ['just', 'like', 'mail', 'bike', 'thanks', 'chip', 'space', 'email']
Topic 8:  ['does', 'know', 'chip', 'like', 'card', 'clipper', 'encryption', 'government']
Topic 9:  ['like', 'card', 'sale', 'video', 'offer', 'jesus', 'good', 'price']
Topic 10:  ['like', 'drive', 'file', 'files', 'sounds', 'program', 'window', 'space']
Topic 11:  ['people', 'like', 'thanks', 'card', 'government', 'windows', 'right', 'think']
Topic 12:  ['think', 'good', 'thanks', 'need', 'chip', 'know', 'really', 'bike']
Topic 13:  ['think', 'does', 'just', 'mail', 'like', 'game', 'file', 'chip']
Topic 14:  ['know', 'good', 'people', 'windows', 'file', 'sale', 'files', 'price']
Topic 15:  ['space', 'know', 'think', 'nasa', 'card', 'year', 'shuttle', 'article']
Topic 16:  ['does', 'israel', 'think', 'right', 'israeli', 'sale', 'jews', 'window']
Topic 17:  ['good', 'space', 'card', 'does', 'thanks', 'year', 'like', 'nasa']
Topic 18:  ['people', 'does', 'window', 'problem', 'space', 'using', 'work', 'server']
Topic 19:  ['right', 'bike', 'time', 'windows', 'space', 'does', 'file', 'thanks']
Topic 20:  ['file', 'problem', 'files', 'need', 'time', 'card', 'game', 'people']
'''

# 잠재 의미 분석(LSA)은 쉽고 빠르게 구현이 가능
# 하지만 문서에 포함된 단어가 가우시안 분포를 따라야만 LSA를 적용할 수 있음
# 문서가 업데이트 된다면 처음부터 다시 SVD를 적용해줘야 하므로 자원이 많이 소모됨
