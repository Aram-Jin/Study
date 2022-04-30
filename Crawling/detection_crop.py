import os, shutil, numpy as np, pandas as pd, time, urllib.request, warnings, cv2
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
warnings.filterwarnings(action='ignore')

print('----------------------------crop을 시작합니다.-----------------------------')
    
cascade = cv2.CascadeClassifier('load_weights/haarcascade_frontalface_alt.xml')      # 가중치 로드
    
img_name = os.listdir(f'data/{saveword}/original')
errorcode = []  

for i,j in enumerate(img_name,start=1):
    img = cv2.imread(f'data/{saveword}/original/{j}')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=(20,20))
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        faces = cv2.resize(faces,dsize=(1024,1024), interpolation=cv2.INTER_LINEAR)   # 저장할 사진 size입력.
        try:
            cv2.imwrite(f'data/{saveword}/crop/{j}', faces)
        except:
            errorcode.append(f'{i}번째 사진 crop에러.')
        print(f'{i}장째 완료.')


