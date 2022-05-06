from matplotlib.pyplot import axis, text
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import urllib.request,time,warnings,os     # url주소,경고창,os폴더생성
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings(action='ignore')    # 경고 무시

# 크롬창에서 F12누르면 html 작업창이 나옴.
# 크롬드라이버 옵션 설정, 2번째줄은 보안관련 해제해주는 옵션
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('C://chromedriver.exe', options=options)


driver.get("https://finance.naver.com/")           
time.sleep(0.2)

USD = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[1]/div[1]/table/tbody/tr[1]/th/a').text
time.sleep(0.2)
price1 = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[1]/div[1]/table/tbody/tr[1]/td[1]').text.replace(",", "")
time.sleep(0.2)

USDINDEX = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[1]/div[2]/h2').text
time.sleep(0.2)
price11 = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[1]/div[2]/table/tbody/tr[4]/td[1]').text.replace(",", "")
time.sleep(0.2)

WTI = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[2]/div[1]/h2').text
time.sleep(0.2)
price2 = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[2]/div[1]/table/tbody/tr[2]/td[1]').text.replace(",", "")
time.sleep(0.2)

GOLD = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[2]/div[2]/h2').text
time.sleep(0.2)
price3 = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]').text.replace(",", "")
time.sleep(0.2)

NASDAC = driver.find_element_by_xpath('//*[@id="container"]/div[2]/div/div[1]/h3').text
time.sleep(0.2)
price4 = driver.find_element_by_xpath('//*[@id="container"]/div[2]/div/div[1]/table/tbody/tr[1]/td[1]').text.replace(",", "")
time.sleep(0.2)

{USD : price1, USDINDEX : price11, WTI : price2, GOLD : price3, NASDAC: price4}
dic = [price1, price11, price2, price3, price4]
  
def x_data(code):
    
    driver.get(f"https://finance.naver.com/item/main.naver?code={code}")

    PRICE = driver.find_element_by_xpath('//*[@id="middle"]/div[1]/div[1]/h2/a').text
    time.sleep(0.2)
    price21 = driver.find_element_by_xpath('//*[@id="chart_area"]/div[1]/div/p[1]').text.replace("\n", "").replace(",", "")
    time.sleep(0.2)

    VOLUME = driver.find_element_by_xpath('//*[@id="chart_area"]/div[1]/table/tbody/tr[1]/td[3]/span').text
    time.sleep(0.2)
    price22 = driver.find_element_by_xpath('//*[@id="chart_area"]/div[1]/table/tbody/tr[1]/td[3]/em').text.replace("\n", "").replace(",", "")
    time.sleep(0.2)

    target = driver.find_element_by_xpath('//*[@id="chart_area"]/div[1]/div/p[2]/span[1]').text
    time.sleep(0.2)
    updown = driver.find_element_by_xpath('//*[@id="chart_area"]/div[1]/div/p[2]/em[2]/span[1]').text.replace("\n", "")
    time.sleep(0.2)
    
    {PRICE : price21, VOLUME : price22, target : updown}
    dic2 = [price21,price22,updown]  
  
    return dic2

codes = ['090710', '060230', '065440', '214680', '035460', '038110', '065420', '058450', '119850', '019550', '048830', '027710', '049080', '900260' ]  # 종목코드 담기
x=[]
for code in codes:
    dic + x_data(code)
    x.append(dic + x_data(code)) 
# print(x)

names = ['USD', 'USDINDEX', 'WTI', 'GOLD', 'NASDAC','PRICE','VOLUME','target']

datasets = pd.DataFrame(x, columns=names, index=codes)
# datasets = pd.to_numeric(datasets, errors='ignore')
#datasets.index.name = "code"

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
label = datasets['target']
le.fit(label)
datasets['target'] = le.transform(label)
print(datasets)

#1. 데이터
x1 = datasets.drop(['target'], axis=1)
y1 =datasets['target']

x1['USD'] = pd.to_numeric(x1['USD'])
x1['USDINDEX'] = pd.to_numeric(x1['USDINDEX'])
x1['WTI'] = pd.to_numeric(x1['WTI'])
x1['GOLD'] = pd.to_numeric(x1['GOLD'])
x1['NASDAC'] = pd.to_numeric(x1['NASDAC'])
x1['PRICE'] = pd.to_numeric(x1['PRICE'])
x1['VOLUME'] = pd.to_numeric(x1['VOLUME'])

print(x1.shape)  
print(y1.shape)  
print(np.unique(y1))  # ['+' '-']
# print(y1)
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
from tensorflow.keras.models import Model
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

model1 = XGBClassifier()
model2 = CatBoostClassifier()

voting_model = VotingClassifier(
    estimators=[('xg', model1),('cat', model2)],
    voting='soft')

# 앙상블 모델 학습
voting_model.fit(x_train, y_train)

# 모델 비교
for model in (model1,model2,voting_model):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(model.__class__.__name__," : ", accuracy_score(y_test,y_pred))

# CatBoostClassifier  :  0.6666666666666666
# VotingClassifier  :  0.6666666666666666
