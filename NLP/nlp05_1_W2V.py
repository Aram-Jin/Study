import re
import urllib.request
import zipfile
from lxml import etree
from nltk import word_tokenize, sent_tokenize

### 1) 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")


### 2) 훈련 데이터 전처리하기
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]

print('총 샘플의 개수 : {}'.format(len(result)))  # 총 샘플의 개수 : 273424

for line in result[:3]:
    print(line)
    
    
### 3) Word2Vec 훈련시키기    
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
'''
vector_size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
window = 컨텍스트 윈도우 크기
min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
workers = 학습을 위한 프로세스 수
sg = 0은 CBOW, 1은 Skip-gram.
'''
model_result = model.wv.most_similar("man")
print(model_result)
# [('woman', 0.8365211486816406), ('guy', 0.7995544075965881), ('boy', 0.7500938773155212), ('lady', 0.7410575151443481), ('soldier', 0.7401436567306519), ('girl', 
# 0.719757616519928), ('gentleman', 0.7119170427322388), ('kid', 0.6930317878723145), ('poet', 0.6568674445152283), ('david', 0.6557236909866333)]


### 4) Word2Vec 모델 저장하고 로드하기
model.wv.save_word2vec_format('eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드

model_result = loaded_model.most_similar("man")
print(model_result)
# [('woman', 0.8449808359146118), ('guy', 0.8157075643539429), ('lady', 0.7871572375297546), ('boy', 0.7587407827377319), ('gentleman', 0.7372528910636902), 
#  ('girl', 0.7316393256187439), ('soldier', 0.7263344526290894), ('poet', 0.7149562835693359), ('kid', 0.6815341114997864), ('surgeon', 0.6727480888366699)]