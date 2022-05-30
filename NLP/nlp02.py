import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
nltk.download('punkt')

text_sample = 'The Mattrix is everywhere its all around us, here even in this room. You can see it out your window or on your television. You feel it when you go to work,or go to church or pay your taxes.'
sentences = sent_tokenize(text = text_sample)

sentence = "The Mattrix is everywhere its all around us, here even in this room"
words = word_tokenize(sentence)

def tokenize_text(text):
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

word_tokens = tokenize_text(text_sample) 
print(word_tokens)

# stop word 제거
nltk.download('stopwords')

# print('영어 stop words 개수:', len(nltk.corpus.stopwords.words('english')))  # 영어 stop words 개수: 179
# print(nltk.corpus.stopwords.words('english')[:20])
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his']
# 영어에 대한 stopword는 179개이고 다음과 같다.


stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
# 위 예제에서 3개의 문장별로 얻은 word_tokens list에 대해 스톱워드를 제거하는 반복문
for sentence in word_tokens:
    filtered_words =[]
    # 개별 문장별로 토큰화된 문장 list에 대해 스톱워드를 제거하는 반복문
    for word in sentence:
        # 소문자로 모두 반환합니다.
        word = word.lower()
        # 토큰화된 개별 단어가 스톱워드의 단어에 포함되지 않으면 word_tokens에 추가
        if word not in stopwords:
            filtered_words.append(word)
    all_tokens.append(filtered_words)
        
print(all_tokens)
# [['mattrix', 'everywhere', 'around', 'us', ',', 'even', 'room', '.'], ['see', 'window', 'television', '.'], 
# ['feel', 'go', 'work', ',', 'go', 'church', 'pay', 'taxes', '.']]
    
