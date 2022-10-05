# python lib
import os, sys, logging, json, random, datetime
from time import sleep
from urllib import parse
from itertools import islice


# lib
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import requests

# selenium lib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.remote_connection import LOGGER as seleniumLogger # Set the threshold for selenium to WARNING
from urllib3.connectionpool import log as urllibLogger # Set the threshold for urllib3 to WARNING
from webdriver_manager.chrome import ChromeDriverManager

# config 설정
def get_config() -> dict:  
    # getting config value
    config_json = None
    try:
        with open('config.json', 'rb') as config_file:
            config_json = json.load(config_file)
    except Exception as e:
        print(f"채널톡 크롤링 setting config ERROR: {e}, {type(e).__name__}, {type(e)}")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)    

    # none이면 file 내용이 비어 있음 
    if config_json is None: 
        print("채널톡 크롤링 setting config ERROR: 세팅 값이 NONE이 될 수 없습니다.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    return config_json



def set_driver_config():
    '''
    - selenium 의 webdriver 설정
    - keep track of what happens in their software applications or online services.
    '''
    seleniumLogger.setLevel(logging.CRITICAL) 
    urllibLogger.setLevel(logging.CRITICAL)
    ua = UserAgent(verify_ssl=False)
    webdriver_options = webdriver.ChromeOptions()

    # 아래 옵션들을 주석 처리 하면, 실제 브라우저 띄워져서 로그인 - 수신함 까지 가는 시뮬레이션 브라우저를 볼 수 있음
    # 하지만, 크롤링 초기 속도가 저하될 수 있음 
    webdriver_options.add_argument("--headless")    # headless
    webdriver_options.add_argument('--window-size=1920x1080')   # 혹시나 mobile로 인식해 issue 생길 염려 제거
    webdriver_options.add_argument("--disable-gpu") # gpu 가동 X -> image 파일 등의 렌더링을 막아 로딩을 최소화 시켜줌
    webdriver_options.add_argument('--log-level=3') # 불필요한 로깅 하지 않음 


    webdriver_options.add_argument(ua.random)       # 사람 처럼 인식시켜주기 위해 user agent 값을 바꿔줌
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=webdriver_options)    
    return driver   

def click_dom(driver, xpath: str):  
    driver.find_element(By.XPATH, xpath).click()
    
    

if __name__ == "__main__":
    config_json: dict = get_config()     
    # driver config and getting driver
    driver = set_driver_config()
    driver.implicitly_wait(3)
    driver.get(config_json['sign_up_url'])
    
    
    # login
    sign_in_email = driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/div/div/main/div/div/div/form/div[1]/input')
    sign_in_email.send_keys(config_json["id"])
    sign_in_pass = driver.find_element(By.XPATH, '//*[@id="appGlobalContentWrapper"]/div/main/div/div/div/form/div[2]/input')
    sign_in_pass.send_keys(config_json["pass"])
    click_dom(driver, "/html/body/div[1]/div[3]/div/div/main/div/div/div/form/button")
    sleep(2)
        
    # 채널 선택
    click_dom(driver, "/html/body/div[1]/div[3]/div/div/main/div/div/ul/li[2]")
    
    #수신함(test환경 클릭)     
    click_dom(driver, "/html/body/div[1]/div[3]/div/div/div/div/div/div/div[1]/div[1]/a[2]")
    sleep(2)
        
    # ====================requests==================== # 
    channel_id = config_json["channel_id"]
    manager_list_url = config_json["manager_list_url"]
    whole_chat_list = config_json["whole_chat_opened_list"]
    each_manager_init = config_json["each_manager_init"]
    each_chat_init = config_json["detail_chat_init"] 
    each_chat_next = config_json["detail_chat_next"]    
    
    # requests session 쿠키를 셀레늄에 업데이트 하깅
    ua = UserAgent(verify_ssl=False)
    s = requests.Session()
    s.headers.update({"user-agent": ua.random})
    
    # cookie update     
    for cookie in driver.get_cookies():
        s.cookies.update({ cookie["name"]: cookie["value"]})

    # dict_keys(['messages', 'sessions', 'userChats', 'users', 'managers', 'chatTags'])

    whole_chat_list = s.get(whole_chat_list.format(channel_id)).json()
    message_list = whole_chat_list["messages"]     
    userchats_list = whole_chat_list["userChats"]
    uesrs_list = whole_chat_list["users"]
    managers_list = whole_chat_list["managers"]
    for managers in managers_list:
        manager_Id = managers["id"]
        manager_name = managers["name"]
     #   manager_chat_Id = managers["id"]   , manager_chat_Id 
        messge_list = s.get(each_manager_init.format(channel_id)).json()     
        

                
                
                
                
                
            



        

            


    


    
