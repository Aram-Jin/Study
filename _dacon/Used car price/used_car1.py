import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pandas_profiling
import seaborn as sns
import random as rn
import os
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from collections import Counter
from pycaret.regression import *

print("numpy version: {}". format(np.__version__))
print("pandas version: {}". format(pd.__version__))
print("matplotlib version: {}". format(matplotlib.__version__))
print("scikit-learn version: {}". format(sklearn.__version__))
# numpy version: 1.19.5
# pandas version: 1.1.5
# matplotlib version: 3.3.4
# scikit-learn version: 0.23.2

# reproducibility
seed_num = 42 
np.random.seed(seed_num)
rn.seed(seed_num)
os.environ['PYTHONHASHSEED']=str(seed_num)

train = pd.read_csv('./data/user_data/train.csv')
test = pd.read_csv('./data/user_data/test.csv')

print(train.shape)  # (1015, 11)
print(train.head())

pr = train.profile_report()
pr.to_file('./data/user_data/pr_report.html')
print(pr)

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


print('정제 전 paint의 unique 카테고리 개수 : ', len(train['paint'].unique()))
temp = clean_text(train['paint']) #메소드 적용
train['paint'] = temp
print('정제 후 paint의 unique 카테고리 개수 : ', len(train['paint'].unique()))

map_list = {i : i for i in np.unique(temp)}

tmp_map = {'off white l':'off white',
            'redl': 'red',
            'gray': 'grey',
            'gery': 'grey',
            'skye blue':'sky blue',
            'sliver':'silver',
            'whine':'white'}

for k in tmp_map.keys():
  v = tmp_map[k]
  map_list[k] = v

train['paint'] = train['paint'].map(map_list)
print(np.unique(train['paint']))


print('정제 전 paint의 unique 카테고리 개수 : ', len(test['paint'].unique()))

temp2 = clean_text(test['paint']) #메소드 적용
test['paint'] = temp2

print('정제 후 paint의 unique 카테고리 개수 : ', len(test['paint'].unique()))

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

train['paint'].value_counts()
print('paint의 unique 카테고리 개수 : ', len(train['paint'].value_counts()))


test_paint = clean_text(test['paint'])
test['paint'] = test_paint
print('test data에서 paint의 unique 카테고리 개수 : ', len(test['paint'].unique()))

test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'black' if x.find('blac') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'gold' if x.find('golf') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

print(test.paint.unique())

train['location'] = train['location'].replace({
    'Abia State' : 'Abia',
    'Abuja ' : 'Abuja',
    'Lagos ' : 'Lagos',
    'Lagos State' : 'Lagos',
    'Ogun State' : 'Ogun'
    })

test['location'] = test['location'].replace({
    'Abuja ' : 'Abuja',
    'Lagos ' : 'Lagos',
    'Lagos State' : 'Lagos',
    'Ogun State' : 'Ogun',
    'Arepo ogun state ' : 'Ogun'
    # Arepo is a populated place located in Ogun State, Nigeria. 출처. 위키백과
})

train['location'] = train['location'].replace({
    'Accra' : 'other',
    'Adamawa ' : 'other',
    'FCT' : 'other',
    'Mushin' : 'other'
})

print(train.location.unique())

test['location'] = test['location'].replace({
    'Accra' : 'other',
    'Adamawa ' : 'other',
    'FCT' : 'other',
    'Mushin' : 'other'
})

print(test.location.unique())

df_train = train.copy()
df_test = test.copy()

fig, ax = plt.subplots(1, 2, figsize=(18,5))
g = sns.histplot(df_train['odometer'], color='b', label='Skewness : {:.2f}'.format(df_train['odometer'].skew()), ax=ax[0])
g.legend(loc='best', prop={'size': 16})
g.set_xlabel("Odometer", fontsize = 16)
g.set_ylabel("Count", fontsize = 16)

g = sns.histplot(df_train['year'], color='b', label='Skewness : {:.2f}'.format(df_train['year'].skew()), ax=ax[1])
g.legend(loc='best', prop={'size': 16})
g.set_xlabel("Year", fontsize = 16)
g.set_ylabel("Count", fontsize = 16)
plt.show()

numeric_fts = ['odometer', 'year']
outlier_ind = []
for i in numeric_fts:
  Q1 = np.percentile(df_train[i],25)
  Q3 = np.percentile(df_train[i],75)
  IQR = Q3-Q1
  outlier_list = df_train[(df_train[i] < Q1 - IQR * 1.5) | (df_train[i] > Q3 + IQR * 1.5)].index
  outlier_ind.extend(outlier_list)
  
# Drop outliers
train_df = df_train.drop(outlier_ind, axis = 0).reset_index(drop = True)
print(train_df)

fig, ax = plt.subplots(1, 2, figsize=(18,5))
g = sns.histplot(train_df['odometer'], color='b', label='Skewness : {:.2f}'.format(train_df['odometer'].skew()), ax=ax[0])
g.legend(loc='best', prop={'size': 16})
g.set_xlabel("Odometer", fontsize = 16)
g.set_ylabel("Count", fontsize = 16)

g = sns.histplot(train_df['year'], color='b', label='Skewness : {:.2f}'.format(train_df['year'].skew()), ax=ax[1])
g.legend(loc='best', prop={'size': 16})
g.set_xlabel("Year", fontsize = 16)
g.set_ylabel("Count", fontsize = 16)
plt.show()

print("# outliers to drop :", len(outlier_ind))



cat_fts = ['title', 'location', 'isimported', 'engine', 'transmission', 'fuel', 'paint']

la_train = train_df.copy()

for i in range(len(cat_fts)):
  encoder = LabelEncoder()
  la_train[cat_fts[i]] = encoder.fit_transform(la_train[cat_fts[i]])

plt.figure(figsize = (10,8))
sns.heatmap(la_train[['odometer', 'year', 'paint', 'fuel', 'transmission', 'engine', 'target']].corr(), annot=True)
plt.show()

print(train_df['title'].unique()[:20])

train_df['company'] = train_df['title'].apply(lambda x : x.split(" ")[0])
df_test['company'] = df_test['title'].apply(lambda x : x.split(" ")[0])

print(train_df['company'].unique())
print("#fts :", len(train_df['company'].unique()), '\n')
print(df_test['company'].unique())
print("#fts :", len(df_test['company'].unique()), '\n')

cat_fts2 = ['title', 'location', 'isimported', 'engine', 'transmission', 'fuel', 'paint', 'company']

for i in range(len(cat_fts2)):
    print(cat_fts2[i], ":")
    print(train_df[cat_fts2[i]].unique())
    print("#fts :", len(train_df[cat_fts2[i]].unique()), '\n')
    
for i in range(len(cat_fts2)):
    print(cat_fts2[i], ":")
    print(df_test[cat_fts2[i]].unique())
    print("#fts :", len(df_test[cat_fts2[i]].unique()), '\n')
    
train_data = train_df.copy()
test_data = df_test.copy()

for i in range(len(cat_fts2)):
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse = False)

    transformed = onehot_encoder.fit_transform(train_data[cat_fts2[i]].to_numpy().reshape(-1, 1))
    onehot_df = pd.DataFrame(transformed, columns=onehot_encoder.get_feature_names())
    train_data = pd.concat([train_data, onehot_df], axis=1).drop(cat_fts2[i], axis=1)

    test_transformed = onehot_encoder.transform(test_data[cat_fts2[i]].to_numpy().reshape(-1, 1))
    test_onehot_df = pd.DataFrame(test_transformed, columns=onehot_encoder.get_feature_names())
    test_data = pd.concat([test_data, test_onehot_df], axis=1).drop(cat_fts2[i], axis=1)
    
print(train_data.columns)
print(test_data.columns)

train_x = train_data.drop('id', axis = 1)
test_x = test_data.drop('id', axis = 1)

print(train_x.shape)
print(test_x.shape)

py_reg = setup(train_x, target = 'target', session_id = seed_num, silent = True)

print(compare_models())

catboost = create_model('catboost', verbose = False)
rf = create_model('rf', verbose = False)
gbr = create_model('gbr', verbose = False)

blended_model = blend_models(estimator_list = [catboost, rf, gbr])

final_model = finalize_model(blended_model)
prediction = predict_model(final_model, data = test_x)

pred = prediction['Label']

# 제출용 sample 파일을 불러옵니다.
submission = pd.read_csv('./data/user_data/sample_submission.csv')
submission.head()

# 위에서 구한 예측값을 그대로 넣어줍니다.
submission['target'] = pred

# 데이터가 잘 들어갔는지 확인합니다.
submission.head()

submission.to_csv('./data/user_data/submit12.csv', index=False)
