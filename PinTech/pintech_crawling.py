# selenium 관련 import
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import urllib.request,time,warnings,os     # url주소,경고창,os폴더생성

warnings.filterwarnings(action='ignore')    # 경고 무시

# 크롬창에서 F12누르면 html 작업창이 나옴.
# 크롬드라이버 옵션 설정, 2번째줄은 보안관련 해제해주는 옵션
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('C://chromedriver.exe', options=options)

# 크롬실행 후 검색창에 keyword 입력
driver.get("https://finance.naver.com/sise/")           # 실행할 창 주소 넣어주세요.
driver.find_element_by_xpath('/html/body/div[2]/div[2]/div/div/ul/li[2]/a/span').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="newarea"]/div[1]/ul/li[1]/ul/li[20]/a').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option2"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option8"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option4"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option6"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option12"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option7"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="option9"]').click()
time.sleep(0.5)
driver.find_element_by_xpath('//*[@id="contentarea_left"]/div[2]/form/div/div/div/a[1]/img').click()
time.sleep(5)
