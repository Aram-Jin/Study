import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import VotingClassifier

import joblib

from lightgbm import LGBMClassifier
from lightgbm import plot_importance

from catboost import CatBoostClassifier

path = "../_data/dacon/Jobcare_data/" 
SUBMIT_PATH = "../_data/dacon/Jobcare_data/" 

train_data = pd.read_csv(path + 'train.csv')
print(train_data.info())
test_data = pd.read_csv(path + 'test.csv')
print(train_data.head())

d_l_matched_count = 0
d_m_matched_count = 0
d_s_matched_count = 0

h_l_matched_count = 0
h_m_matched_count = 0
h_s_matched_count = 0

for i in tqdm(range(501951)):
    if train_data['d_l_match_yn'][i] == True:
        d_l_matched_count += 1
    if train_data['d_m_match_yn'][i] == True:
        d_m_matched_count += 1
    if train_data['d_s_match_yn'][i] == True:
        d_s_matched_count += 1

    if train_data['h_l_match_yn'][i] == True:
        h_l_matched_count += 1
    if train_data['h_m_match_yn'][i] == True:
        h_m_matched_count += 1
    if train_data['h_s_match_yn'][i] == True:
        h_s_matched_count += 1

y = [d_l_matched_count, d_m_matched_count, d_s_matched_count, h_l_matched_count, h_m_matched_count, h_s_matched_count]
x = range(len(y))

label = ['d_l', 'd_m', 'd_s', 'h_l', 'h_m', 'h_s']

colors = ['blue' for i in range(6)]
for i in range(3, 6):
    colors[i] = 'skyblue'

plt.figure(figsize=(10, 10))
plt.bar(x, y, width=0.7, color=colors)
plt.xticks(x, label, fontsize=13)
plt.show()

classification_data = train_data[['d_l_match_yn', 'd_m_match_yn', 'd_s_match_yn', 'h_l_match_yn', 'h_m_match_yn', 'h_s_match_yn']]

plt.figure(figsize=(11, 11))
sns.heatmap(data = classification_data.corr(), annot=True, cmap='YlGnBu')
plt.show()

user_property = train_data[['person_attribute_a', 'person_attribute_a_1', 'person_attribute_b', 'person_prefer_c', 'person_prefer_d_1', 'person_prefer_d_2', 'person_prefer_d_3', 'person_prefer_e', 'person_prefer_h_1', 'person_prefer_h_2', 'person_prefer_h_3']]

plt.figure(figsize=(12, 12))
sns.heatmap(data = user_property.corr(), annot=True, cmap='YlGnBu')
plt.show()

content_column = [
    'contents_attribute_i',
    'contents_attribute_a',
    'contents_attribute_j_1',
    'contents_attribute_j',
    'contents_attribute_c',
    'contents_attribute_k',
    'contents_attribute_l',
    'contents_attribute_d',
    'contents_attribute_m',
    'contents_attribute_e',
    'contents_attribute_h'
]

content_property = train_data[content_column]

plt.figure(figsize=(12, 12))
sns.heatmap(data=content_property.corr(), annot=True, cmap='YlGnBu')
plt.show()

components_count = 2

pca = PCA(n_components = components_count)
pca_components = train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1)

# Nomalization of pca_components
pca_components_scaled = StandardScaler().fit_transform(pca_components)
pca_components_scaled = pca_components_scaled[:1000]

pca_final_components = pca.fit_transform(pca_components_scaled)

target = train_data.target[:1000]
combine_frame = np.c_[pca_final_components, target]

print(combine_frame.shape)

pca_dataframe = pd.DataFrame(data=combine_frame, columns=['PC1', 'PC2', 'target'])

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 10))
plt.scatter(pca_dataframe['PC1'], pca_dataframe['PC2'], c=pca_dataframe['target'])
plt.show()

components_count = 2

tsne = TSNE(n_components=components_count, random_state=0)
tsne_components = train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1)

# Normalization of tsne_components
tsne_components_scaled = StandardScaler().fit_transform(tsne_components)
tsne_components_scaled = tsne_components_scaled[:1000]

tsne_final_copmonents = tsne.fit_transform(tsne_components_scaled)

target = train_data.target[:1000]
combine_frame = np.c_[tsne_final_copmonents, target]
print(combine_frame.shape)

tsne_dataframe = pd.DataFrame(data=combine_frame, columns=['SNE1', 'SNE2', 'target'])

plt.figure(figsize=(10, 10))
plt.scatter(tsne_dataframe['SNE1'], tsne_dataframe['SNE2'], c=tsne_dataframe['target'])
plt.show()

combine_feature_contents_d = train_data['contents_attribute_d'] + train_data['person_prefer_d_1']
train_data['combine_feature_contents_d'] = combine_feature_contents_d

combine_feature_contents_cl = train_data['contents_attribute_c'] + train_data['contents_attribute_l']
train_data['combine_feature_contents_cl'] = combine_feature_contents_cl

combine_feature_contents_ml = train_data['contents_attribute_m'] + train_data['contents_attribute_l']
train_data['combine_feature_contents_ml'] = combine_feature_contents_ml

combine_feature_person_cl = train_data['person_attribute_a_1'] + train_data['person_prefer_e']
train_data['combine_feature_person_ae'] = combine_feature_person_cl

test_data['combine_feature_contents_d'] = combine_feature_contents_d
test_data['combine_feature_contents_cl'] = combine_feature_contents_cl
test_data['combine_feature_contents_ml'] = combine_feature_contents_ml
test_data['combine_feature_person_ae'] = combine_feature_person_cl

lgbm_wrapper = LGBMClassifier(n_estimators=1000, learning_rate=0.01, num_leaves=240, max_depth=120)

target = train_data.target
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1), target, test_size=0.2)

kfold = KFold(n_splits=5, shuffle=True, random_state=52)

cv_accuracy = []
f1_scores = []

for train_index, test_index in kfold.split(x_train):
    x_fold_train, x_fold_test = x_train.iloc[train_index], x_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

    evals = [(x_fold_test, y_fold_test)]
    lgbm_wrapper.fit(x_fold_train, y_fold_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=False)

    preds = lgbm_wrapper.predict(x_fold_test)
    pred_proba = lgbm_wrapper.predict_proba(x_fold_test)[:, 1]

    accuracy = np.round(accuracy_score(y_fold_test, preds), 4)
    cv_accuracy.append(accuracy)

    f1_scores.append(f1_score(y_fold_test, preds))

print(f'cv_accuracy : {cv_accuracy}')
print(f'평균 검증 정확도 : {np.mean(cv_accuracy)}')

print(f'f1_score : {f1_scores}')
print(f'평균 f1 점수 : {np.mean(f1_scores)}')

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm_wrapper, ax=ax)

joblib.dump(lgbm_wrapper, '/content/drive/MyDrive/dacon/jobcare/models/third_model.pkl')

loaded_lgbm = joblib.load('/content/drive/MyDrive/dacon/jobcare/models/third_model.pkl')

preds_proba = lgbm_wrapper.predict_proba(test_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn'], axis=1))
preds = []

for i in range(len(preds_proba)):
    if preds_proba[i][0] >= 0.65:
        preds.append(0)
    else:
        preds.append(1)

submission = pd.read_csv(path + '/jobcare/Jobcare_data/sample_submission.csv')
submission['target'] = preds

submission.to_csv(path + '/jobcare/submission_preds/third_threshold_65.csv', index=False)

params = {
    'iterations': 300,
    'learning_rate': 0.01,
    'depth': 16,
    'eval_metric': 'Logloss',
}

model = CatBoostClassifier(**params)

target = train_data.target
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1), target, test_size=0.2)

kfold = KFold(n_splits=5, shuffle=True, random_state=52)

cv_accuracy = []
f1_scores = []

for train_index, test_index in kfold.split(x_train):
    x_fold_train, x_fold_test = x_train.iloc[train_index], x_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

    evals = [(x_fold_test, y_fold_test)]
    model.fit(x_fold_train, y_fold_train, eval_set=evals, early_stopping_rounds=100)

    preds = model.predict(x_fold_test)
    pred_proba = model.predict_proba(x_fold_test)[:, 1]

    accuracy = np.round(accuracy_score(y_fold_test, preds), 4)
    cv_accuracy.append(accuracy)

    f1_scores.append(f1_score(y_fold_test, preds))

print(f'cv_accuracy : {cv_accuracy}')
print(f'평균 검증 정확도 : {np.mean(cv_accuracy)}')

print(f'f1_score : {f1_scores}')
print(f'평균 f1 점수 : {np.mean(f1_scores)}')

joblib.dump(model, '/content/drive/MyDrive/dacon/jobcare/models/catboost01.pkl')

loaded_catboost = joblib.load('/content/drive/MyDrive/dacon/jobcare/models/catboost01.pkl')

def arithmetic_mean(*values, count):
    sum = float(0)
    for i in values:
        sum += i

    return sum / count

lgbm_preds_proba = loaded_lgbm.predict_proba(train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1))
catboost_preds_proba = loaded_catboost.predict_proba(train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1))

preds = []

for i in range(len(lgbm_preds_proba)):
    if arithmetic_mean(lgbm_preds_proba[i][0], catboost_preds_proba[i][0], count=2) >= 0.65:
        preds.append(0)
    else:
        preds.append(1)
        
mean_f1_score = f1_score(train_data.target, preds)
print(f'f1-score : {mean_f1_score}')

lgbm_preds_proba = loaded_lgbm.predict_proba(test_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn'], axis=1))
catboost_preds_proba = loaded_catboost.predict_proba(test_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn'], axis=1))

preds = []

for i in range(len(lgbm_preds_proba)):
    if arithmetic_mean(lgbm_preds_proba[i][0], catboost_preds_proba[i][0], count=2) >= 0.65:
        preds.append(0)
    else:
        preds.append(1)

submission = pd.read_csv(base_path + '/jobcare/Jobcare_data/sample_submission.csv')
submission['target'] = preds

submission.to_csv(base_path + '/jobcare/submission_preds/arithmetic_mean_threshold_65.csv', index=False)

voting_model = VotingClassifier(estimators=[('lgbm', loaded_lgbm), ('catboost', loaded_catboost)], voting='soft')
voting_model.fit(train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1), train_data.target)

joblib.dump(voting_model, '/content/drive/MyDrive/dacon/jobcare/models/voting_model.pkl')

loaded_voting = joblib.load('/content/drive/MyDrive/dacon/jobcare/models/voting_model.pkl')

voting_preds_proba = loaded_voting.predict_proba(train_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn', 'target'], axis=1))

preds = []

for i in range(len(voting_preds_proba)):
    if voting_preds_proba[i][0] >= 0.65:
        preds.append(0)
    else:
        preds.append(1)

voting_f1_score = f1_score(train_data.target, preds)
print(f'f1-score : {voting_f1_score}')

voting_preds_proba = loaded_voting.predict_proba(test_data.drop(['id', 'contents_open_dt', 'contents_rn', 'person_rn'], axis=1))

preds = []

for i in range(len(voting_preds_proba)):
    if voting_preds_proba[i][0] >= 0.65:
        preds.append(0)
    else:
        preds.append(1)

submission = pd.read_csv(base_path + '/jobcare/Jobcare_data/sample_submission.csv')
submission['target'] = preds

submission.to_csv(base_path + '/jobcare/submission_preds/voting_threshold_65.csv', index=False)
