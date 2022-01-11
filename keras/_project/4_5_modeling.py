import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import csv

# Model list
def models(model):
    if model == 'knn':
        mod = KNeighborsClassifier(2)
    elif model == 'svm':
        mod = SVC(kernel="linear", C=0.025)
    elif model == 'svm2':
        mod = SVC(gamma=2, C=1)
    elif model == 'gaussian':
        mod = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif model == 'tree':
        mod =  DecisionTreeClassifier(max_depth=5)
    elif model == 'forest':
        mod =  RandomForestClassifier(max_depth=0.5, n_estimators=10, max_features=0.5)
    elif model == 'mlp':
        mod = MLPClassifier(alpha=1, max_iter=1000)
    elif model == 'adaboost':
        mod = AdaBoostClassifier()
    elif model == 'gaussianNB':
        mod = GaussianNB()
    elif model == 'qda':
        mod = QuadraticDiscriminantAnalysis()
    return mod

## Data load
data1 = pd.read_csv("./관리종목data.csv")
data2 = pd.read_csv("./안전종목data.csv")
dataset = pd.concat([data1,data2],ignore_index=True)
del dataset['Unnamed: 0']

x = dataset.drop(['Target'], axis=1).to_numpy()
y = dataset['Target'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

#make model list in models function
model_list = ['knn', 'svm', 'svm2', 'gaussian', 'tree', 'forest', 'mlp', 'adaboost', 'gaussianNB', 'qda']

cnt = 0
empty_list = [] #empty list for progress bar in tqdm library
for model in tqdm(model_list, desc = 'Models are training and predicting ... '):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)

    #Training
    clf.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2)  #학습할때는 id와 target을 제외하고 학습! 마지막 column이 라벨이므로 라벨로 설정!

    #Predict
    pred = clf.predict(test_data[:,1:]) #마찬가지로 예측을 할 때에도 id를 제외하고 나머지 feature들로 예측

    #Make answer sheet
    savepath = datapath + 'answers/' #정답지 저장 경로
    with open(savepath + '%s_answer2.csv' % model_list[cnt], 'w', newline='') as f:
        sheet = csv.writer(f)
        sheet.writerow(['id', 'target'])
        for idx, p in enumerate(pred):
            sheet.writerow([idx+1, p])

    cnt += 1