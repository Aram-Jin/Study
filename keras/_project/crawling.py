# requests, beatifulsoup
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook, workbook
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
import time
import urllib.request
import os
import shutil

# wb = Workbook(write_only=True)
# ws = wb.create_sheet('유동비율')
# ws.append(['2017/12','2018/12','2019/12','2020/12','2021/09'])

code = '091970'        
response = requests.get(f'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode={code}&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701')
rate_page = response.text
soup = BeautifulSoup(rate_page, 'html.parser')
for tr_tag in soup.select('tr')[1:]:
    td_tags = tr_tag.select('td')
    row = [
        td_tags[0].get_text(),
        td_tags[1].get_text(),
        td_tags[2].get_text(),
        td_tags[3].get_text(),
        td_tags[4].get_text(),
        td_tags[5].get_text(),        
    ]
print(row)
#     ws.append(row)
    
# # wb.save('나노캠텍유동비율.xlsx')    

        
# requests.get()

import pandas as pd
import requests

fs_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A005930&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701'
fs_page = requests.get(fs_url)
fs_tables = pd.read_html(fs_page.text)