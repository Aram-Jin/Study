from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2.모델
# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

y_predict = model.predict(x_test)


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, classification_report

f1 = f1_score(y_test, y_predict)  #, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')/// average : binary, macro, micro, weighted, None  / zero_division : 1
confusion = confusion_matrix(y_test, y_predict) #, labels=None, sample_weight=None, normalize=None)
accuracy = accuracy_score(y_test, y_predict) #, normalize=True, sample_weight=None)
precision = precision_score(y_test, y_predict) #, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
recall = recall_score(y_test, y_predict) #, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

print('accuracy_score :', accuracy)	
print('recall_score: ', recall)	
print('precision_score: ', precision)	
print('f1_score: ', f1)	
print('오차행렬')
print(confusion)
print('정확도 : {:.4f}\n정밀도 : {:.4f}\n재현율 : {:.4f}'.format(accuracy, precision, recall))

print(classification_report(y_test, y_predict))

roc_curve = roc_curve(y_test, y_predict, pos_label=None, sample_weight=None, drop_intermediate=True)
roc_auc_score = roc_auc_score(y_test, y_predict, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)

print("ROC curve: ", roc_curve)   
print("ROC | AUC Score: ", roc_auc_score)   

