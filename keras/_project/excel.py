# requests, beatifulsoup
import csv
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook, workbook
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from time import sleep
import time
import urllib.request as req
import os
import shutil

wb = Workbook(write_only=True)
ws = wb.create_sheet('재무비율')
ws.append(['2017/12','2018/12','2019/12','2020/12','2021/09'])

response = requests.get('http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701')
rating_page = response.text
soup = BeautifulSoup(rating_page, 'html.parser')

# res = req.urlopen('http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701')
# soup = BeautifulSoup(res, 'html.parser')
# rating = soup.find_all(class_='us_table_ty1 h_fix zigbg_no')
for tr_tag in soup.select('tr', id="p_grid1_1", class_='acd_dep_start_open')[1:]:
    td_tags = tr_tag.select('td')
    row = [
        td_tags[0].get_text(),
        td_tags[1].get_text(),
        td_tags[2].get_text(),
        td_tags[3].get_text(),
        td_tags[4].get_text(),      
    ]
print(row)
