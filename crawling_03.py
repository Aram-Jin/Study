import os
import json
import os
from random import betavariate
from sqlite3 import Row
from unittest import result
import pandas
import pyperclip
import selenium
import sys
import time

from bs4 import *
from selenium.webdriver.common.keys import Keys
from selenium import webdriver

# JSON loading

# 한국은행 데이터 수집 함수

from urllib import response


def get_ECOS_MM(p_flag):
    import urllib
    import urllib.request
    import urllib.parse
    import json
    
    base_url = 'https://ecos.bok.or.kr/jsp/vis/keystat/Key100Stat_n1.jsp'
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Host': 'ecos.bok.or.kr',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Vesion/15.1 Safari/605.1.15',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://ecos.bok.or.kr/jsp/vis/keystat',
        'Connection': 'keep-alive'
        }
    
    # 파라미터 세팅
    params = '?' + urllib.parse.urlencode({
        urllib.parse.quote_plus('languageF1g'): 'MM',
        urllib.parse.quote_plus('languageF1g2'): '1',
        urllib.parse.quote_plus('languageF1g3'): p_flag,
        urllib.parse.quote_plus('languageF1g4'): '%20',
        urllib.parse.quote_plus('languageF1g5'): 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Vesion/15.1 Safari/605.1.15',
    })
    req = urllib.request.Request(base_url + urllib.parse.unqoute(params), headers=headers)
    response_body = urllib.request.urlopen(req).read()
    json_data = json.loads(response_body)[0]
    
    list_row = list()
    for item_info in json_data:
        yyyymm = int(item_info['TIME'])
        if yyyymm < 201401:
            continue
        list_row.append({
            'yyyymm': yyyymm,
            'item_code': p_flag,
            'value': item_info['DATA_VALUE'],
        }) 
    time.sleep(1.5)
    return pandas.DataFrame(list_row)