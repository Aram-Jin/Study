# python lib
from email import message
import os, sys, logging, json, random, datetime
from time import sleep
from urllib import parse
from itertools import islice
import argparse
from xml.dom.minidom import Identified
import pandas as pd 
from datetime import datetime 

# lib
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import requests
import sys

# selenium lib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.remote_connection import LOGGER as seleniumLogger # Set the threshold for selenium to WARNING
from urllib3.connectionpool import log as urllibLogger # Set the threshold for urllib3 to WARNING
from webdriver_manager.chrome import ChromeDriverManager


# config 옵션
def get_config() -> dict: 
    # getting config value
    config_json = None
    try:
        with open("config.json", "rb") as config_file:
            config_json = json.load(config_file)
            
    except Exception as e:
        print(f"채널톡 크롤링 setting config ERROR: {e}, {type(e).__name__}, {type(e)}")
        try:
            sys.exit(0)  
        except SystemExit:
            os._exit(0)  
            
    if config_json is None:
        print(f"채널톡 크롤링 setting config ERROR: 세팅 값이 NONE이 될 수 없습니다.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    return config_json

# 특정 이름 받는 것

class Crawling:
    def __init__(self, target_name): 
        self.chat_Id = Chat_Id(target_name)
   

    
class Chat_Id:
    def __init__(self, target_name):
        self.target_name = target_name     
        self.info_list = []    
    def fetch_chat_ID(self):
        direct_chats = base["directChats"]
        comb = [[]+ data['managerIds'] for data in direct_chats]
        IDs_LIST = list(set([ID for nums in comb for ID in nums]))  # chat_base에 있는 manager ID들
        print(IDs_LIST)           
                                



    

# 셀레늄 옵션 
def set_driver_config():    
    '''
    - selenium 의 webdriver 설정
    - keep track of what happens in their software applications or online services.
    '''
    seleniumLogger.setLevel(logging.CRITICAL) 
    urllibLogger.setLevel(logging.CRITICAL)
    ua = UserAgent(verify_ssl=False)
    webdriver_options = webdriver.ChromeOptions()  

    webdriver_options.add_argument("--headless")    # headless
    webdriver_options.add_argument('--window-size=1920x1080')   # 혹시나 mobile로 인식해 issue 생길 염려 제거
    webdriver_options.add_argument("--disable-gpu") # gpu 가동 X -> image 파일 등의 렌더링을 막아 로딩을 최소화 시켜줌
    webdriver_options.add_argument('--log-level=3') # 불필요한 로깅 하지 않음 


    webdriver_options.add_argument(ua.random)       # 사람 처럼 인식시켜주기 위해 user agent 값을 바꿔줌
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=webdriver_options)    
    return driver


# 셀레늄 실행 함수 
def click_dom(driver, xpath: str):
    driver.find_element(By.XPATH, xpath).click()       
    
   
# list ==> dictionary 
def convert(lst: list):
    result = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return result  



# 결과를 깔끔하게? 나오게 하는 함수
def cleaned_output(result: list):
    text_list = []
    output = {}
    for i in result:
        for x in i:
            part = x.split(",")
            ID, username, name, message = part[0], part[1], part[2], part[3]
            text = message.split(":")[1] # 숫자만         

            value = username.split(":")+name.split(":")+message.split(":")
            dic_value = convert(value) 
            if ID not in output:
                output[ID] = dic_value   
                output[ID]["메세지"] = [output[ID]["메세지"]]
            else:
                text_list.append(text)
                output[ID]["메세지"].append(text)
    return output 
    
    
if __name__ == "__main__":
    config_json = dict = get_config()
    driver = set_driver_config()
    driver.implicitly_wait(3)
    driver.get(config_json['sign_up_url'])  
    
    # login 
    email = driver.find_element(By.XPATH,'//*[@id="appGlobalContentWrapper"]/div/main/div/div/div/form/div[1]/input')
    email.send_keys(config_json["id"])
    pwd = driver.find_element(By.XPATH,'//*[@id="appGlobalContentWrapper"]/div/main/div/div/div/form/div[2]/input')
    pwd.send_keys(config_json["pass"])
    click_dom(driver, '//*[@id="appGlobalContentWrapper"]/div/main/div/div/div/form/button')
    sleep(2)   
    
    
    # 팀챗 클릭    
    click_dom(driver, "/html/body/div[1]/div[3]/div/div/div/div/div/div/div[1]/div[1]/a[1]/div")
    sleep(1)  
    
    
     
    # ====================requests==================== # 
    channel_id = config_json["channel_id"]
    manager_list_url = config_json["manager_list_url"]
    direct_chat_base = config_json["direct_chat_base"]
    direct_msg_init = config_json["direct_msg_init"]
    direct_msg_next = config_json["direct_msg_next"] 
    
    
    # requests session 쿠키를 셀레늄에 업데이트 하깅
    ua = UserAgent(verify_ssl=False)
    s = requests.Session()
    s.headers.update({"user-agent": ua.random})
 
    # cookie update     
    for cookie in driver.get_cookies():
        s.cookies.update({ cookie["name"]: cookie["value"]})
                
    # 실행
    base = s.get(config_json["direct_chat_base"].format(channel_id)).json() 
    manager_list = s.get(config_json["manager_list_url"].format(channel_id)).json()
   
   
   
    direct_chats = base["directChats"]
    comb = [[]+ data['managerIds'] for data in direct_chats]
    IDs_LIST = list(set([ID for nums in comb for ID in nums]))  # chat_base에 있는 manager ID들
    comb = [[]+ data['managerIds'] for data in direct_chats]
        
    Manager_Id_list = {data["id"]:data["name"] for data in manager_list["managers"]}
    print(Manager_Id_list)
    result = []
    result2 = []
    for k, v in Manager_Id_list.items():
        if k not in IDs_LIST:
            result.append([k,v])
        else:
            result2.append([k,v])
    print("=============chatbase에 없는 ID================")
    print(result)
    print("=============chatbase에 있는 ID================")
    print(result2)
      