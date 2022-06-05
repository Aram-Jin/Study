import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt   
 
# 데이터
train = pd.read_csv('../data/user_data/train.csv')
test = pd.read_csv('../data/user_data/test.csv')

# print(f'train data set은 {train.shape[1]} 개의 feature를 가진 {train.shape[0]} 개의 데이터 샘플로 이루어져 있습니다.')
# train data set은 11 개의 feature를 가진 1015 개의 데이터 샘플로 이루어져 있습니다.

# 데이터의 최상단 5 줄을 표시합니다.
print(train.head())
# print(train.shape, test.shape)   # (1015, 11) (436, 10)

# 결측치 
def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)

# train.info()
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   id            1015 non-null   int64
#  1   title         1015 non-null   object
#  2   odometer      1015 non-null   int64
#  3   location      1015 non-null   object
#  4   isimported    1015 non-null   object
#  5   engine        1015 non-null   object
#  6   transmission  1015 non-null   object
#  7   fuel          1015 non-null   object
#  8   paint         1015 non-null   object
#  9   year          1015 non-null   int64
#  10  target        1015 non-null   int64
# dtypes: int64(4), object(7)

plt.figure(figsize=(10,10))
plt.hist(train['target'], bins=50)
plt.title('Target Histogram')
plt.show()

# target 데이터의 분포가 쏠림현상이 심해 로그변환
log_target = np.log(train['target'])

plt.figure(figsize=(10,10))
plt.hist(log_target, bins=50)
plt.title('Target Histogram with logarithm')
plt.show()

# 수치형 변수 히스토그램 확인: odometer와 year
plt.style.use("ggplot")

plt.figure(figsize=(12,6))
plt.suptitle("Histogram", fontsize=20)

plt.subplot(1,2,1)
plt.hist(train.odometer, bins=50)
plt.title('Odometer Histogram')

plt.subplot(1,2,2)
plt.hist(train.year, bins=50)
plt.title('Year Histogram')
plt.show()

# 1900년도 이전 데이터 확인
print(train[train['year'] < 1900])
#       id                title  odometer location    isimported          engine transmission    fuel  paint  year    target
# 415  415  Mercedes-Benz ATEGO    403461    Lagos  Locally used  4-cylinder(I4)       manual  diesel  white  1218   6015000  -> year가 이상해 데이터 삭제
# 827  827     Mercedes-Benz/52    510053    Lagos  Locally used  6-cylinder(V6)       manual  diesel  white  1217  75015000

train = train[train['year'] > 1900]
print(train.shape) # (1013, 11)

train = train.drop('id', axis = 1).reset_index().drop('index', axis = 1).reset_index().rename({'index':'id'}, axis = 'columns')


data_description = train.describe().iloc[:,1:3]
print(data_description)

# 데이터 분포를 히스토그램에 평균(빨강)과 중앙값(초록)을 선으로 표시
plt.style.use("ggplot")

plt.figure(figsize=(12,6))
plt.suptitle("Histogram", fontsize=20)

plt.subplot(1,2,1)
plt.hist(train['odometer'], bins=50, color='#eaa18a', edgecolor='#7bcabf')
plt.title('odometer')
plt.axvline(data_description['odometer']['mean'], c='#f55354', label = f"mean = {round(data_description['odometer']['mean'], 2)}")
plt.axvline(data_description['odometer']['50%'], c='#518d7d', label = f"median = {round(data_description['odometer']['50%'], 2)}")

plt.subplot(1,2,2)
# 수치형 데이터 통계치 그래프
plt.hist(train['year'], bins = 50, color='#eaa18a', edgecolor='#7bcabf')
plt.title('year')
plt.axvline(data_description['year']['mean'], c='#f55354', label = f"mean = {round(data_description['year']['mean'], 2)}")
plt.axvline(data_description['year']['50%'], c='#518d7d', label = f"median = {round(data_description['year']['50%'], 2)}")
plt.show()

print('Odometer 평균은', round(data_description['odometer']['mean']), '입니다')  # Odometer 평균은 116171 입니다
print('Odometer 중앙값은', round(data_description['odometer']['50%']), '입니다')  # Odometer 중앙값은 94803 입니다

print('Year 평균은', round(data_description['year']['mean']), '입니다')  # Year 평균은 2010 입니다
print('Year 중앙값은', round(data_description['year']['50%']), '입니다')  # Year 중앙값은 2010 입니다

# target과 수치형데이터의 상관관계 분석
import seaborn as sns

plt.style.use("ggplot")

plt.figure(figsize=(12,6))
plt.suptitle("Histogram", fontsize=20)

plt.subplot(1,2,1)
sns.regplot(x='odometer', y='target', data=train,  color='#eaa18a', line_kws=  {'color': '#f55354'})
plt.title('odometer')

plt.subplot(1,2,2)
sns.regplot(x='year', y='target', data=train,  color='#eaa18a', line_kws=  {'color': '#f55354'})
plt.title('year')
plt.show()

from sklearn.preprocessing import MinMaxScaler

# 수치형 데이터 상관관계 히트맵 시각화
train_corr = train[['odometer', 'year', 'target']]
scaler= MinMaxScaler() 
train_corr[train_corr.columns] = scaler.fit_transform(train_corr[train_corr.columns])
corr28 = train_corr.corr(method= 'pearson')

plt.figure(figsize=(12,10))
sns.heatmap(data = corr28, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.title('Correlation between features', fontsize=30)
plt.show()


# Target과 피쳐들의 상관관계
s28 = corr28.unstack()
df_temp28 = pd.DataFrame(s28['target'].sort_values(ascending=False), columns=['target'])
df_temp28.style.background_gradient(cmap='viridis')


# print(train.describe(include="object"))
#                title location    isimported          engine transmission    fuel  paint
# count           1013     1013          1013            1013         1013    1013   1013
# unique           201       13             3               8            2       2     76
# top     Toyota Camry    Lagos  Foreign Used  4-cylinder(I4)    automatic  petrol  Black
# freq             129      790           744             617          965     970    292

# print(train.describe(include="object").columns) 
# Index(['title', 'location', 'isimported', 'engine', 'transmission', 'fuel', 'paint'], dtype='object')


# 파생변수 생성
print(train['title'].value_counts()[:20])

train['title'].apply(lambda x : x.split(" ")[0])

train['brand'] = train['title'].apply(lambda x : x.split(" ")[0])
print(train.head())

print(train['brand'].value_counts().head())

print('title의 unique 카테고리 개수 : ', len(train['title'].value_counts()))  # title의 unique 카테고리 개수 :  201
print('brand의 unique 카테고리 개수 : ', len(train['brand'].value_counts()))  # brand의 unique 카테고리 개수 :  41

print(train['paint'].value_counts()[:20])

import re 

def clean_text(texts): 
    corpus = [] 
    for i in range(0, len(texts)): 
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>\<]', '',texts[i]) #@%*=()/+ 와 같은 문장부호 제거
        review = re.sub(r'\d+','',review)#숫자 제거
        review = review.lower() #소문자 변환
        review = re.sub(r'\s+', ' ', review) #extra space 제거
        review = re.sub(r'<[^>]+>','',review) #Html tags 제거
        review = re.sub(r'\s+', ' ', review) #spaces 제거
        review = re.sub(r"^\s+", '', review) #space from start 제거
        review = re.sub(r'\s+$', '', review) #space from the end 제거
        review = re.sub(r'_', ' ', review) #space from the end 제거
        #review = re.sub(r'l', '', review)
        corpus.append(review) 
        
    return corpus

temp = clean_text(train['paint']) #메소드 적용
train['paint'] = temp

print('brand의 unique 카테고리 개수 : ', len(train['paint'].unique()))  # brand의 unique 카테고리 개수 :  51

print(train['paint'].value_counts()[:20])

# 오타 및 색상면 수정
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

print(train['paint'].value_counts())
print('paint의 unique 카테고리 개수 : ', len(train['paint'].value_counts())) # paint의 unique 카테고리 개수 :  18


plt.style.use("ggplot")

plt.figure(figsize=(25,15))
count = 1

for i in train.describe(include="object").columns:
    plt.subplot(4,2,count)
    # countplot 을 사용해서 데이터의 분포를 살펴봅니다.
    sns.countplot(data=train, x=i)
    count += 1
    
    
train_title10 = train[train['title'].apply(lambda x : x in train['title'].value_counts()[:10].keys())]
train_brand10 = train[train['brand'].apply(lambda x : x in train['brand'].value_counts()[:10].keys())]    
    
plt.style.use("ggplot")

plt.figure(figsize=(25,15))
plt.subplot(4,2,1)
sns.countplot(data=train_title10, x='title')
plt.subplot(4,2,2)
sns.countplot(data=train_brand10, x='brand')
count = 3

for i in train.describe(include="object").columns.drop(['title'])[:-1]:
    plt.subplot(4,2,count)
    sns.countplot(data=train, x=i)
    count += 1    

plt.style.use("ggplot")

plt.figure(figsize=(25,15))
plt.subplot(4,2,1)
sns.violinplot(data=train_title10, x='title', y ='target')
plt.subplot(4,2,2)
sns.violinplot(data=train_brand10, x='brand', y ='target')
count = 3

for i in train.describe(include="object").columns.drop(['title'])[:-1]:
    plt.subplot(4,2,count)
    sns.violinplot(data=train, x=i, y ='target')
    count += 1
