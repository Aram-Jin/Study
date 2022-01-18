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
'''
res = req.urlopen('http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701')
soup = BeautifulSoup(res, 'html.parser')
rating = soup.find_all(class_='us_table_ty1 h_fix zigbg_no')
for tr_tag in soup.select('tr', id="p_grid1_1", class_='acd_dep_start_open')[1:]:
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
'''
# code = '091970'   
# response = requests.get(f'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode={code}&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701')

# # res = req.urlopen('http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701')

# soup = BeautifulSoup(response, 'html.parser')
# print(soup.find_all(class_='ul_col2wrap pd_t25'))
 
# for tag in soup.select('tr')[1:]:


# wb = Workbook(write_only=True)
# ws = wb.create_sheet('유동비율')
# ws.append(['2017/12','2018/12','2019/12','2020/12','2021/09'])

# code = '091970'        
# response = requests.get(f'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode={code}&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701')
# rate_page = response.text

# for tr_tag in rating.select('tr'):
#     td_tags = tr_tag.select('td')

# print(rating)

# rating1 = rating.find_all(id="p_grid1_1", class_='acd_dep_start_open')

# #     ws.append(row)
    
# # # wb.save('나노캠텍유동비율.xlsx')    

        
# # requests.get()

# import pandas as pd
# import requests

# fs_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A005930&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701'
# fs_page = requests.get(fs_url)
# fs_tables = pd.read_html(fs_page.text)


code = '091970'  
# response = requests.get(f'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode={code}&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701')

res = req.urlopen('http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A'+code+'&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701')
# data = res.json( )
soup = BeautifulSoup(res, 'html.parser')

fin_info = soup.find_all(class_='us_table_ty1 h_fix zigbg_no')
fin_data = [item.get_text().strip() for item in fin_info.select('td')]
print(fin_data)

# col_data = [item.get_text() for item in fin_info.find_all('td class') ]
# data = soup.find_all(class_='us_table_ty1 h_fix zigbg_no')
# # data.find_all('td',{'class':'r'})

# ratings1 = soup.select_one("#p_grid1_1")
# ratings2 = soup.select_one("#p_grid1_2")
# ratings3 = soup.select_one("#p_grid1_3")
# ratings4 = soup.select_one("#p_grid1_4")

#p_grid1_4
# for rating in ratings:
#     per = rating.select_one('')

# trs = data.select('table > tbody > tr')
# print(trs)


# rating = soup.find('tr', attrs = {'class':'acd_dep_start_open'})
# print(soup.td.attrs)
# print(soup.find('tr', attrs={ 'class':'r'})) 
# print(rating)
# rating2 = soup.find(class_='us_table_ty1 h_fix zigbg_no')
# rating2_1 = soup.find(id='p_grid1_1',class_='acd_dep_start_open')

# ratomg3 = soup.find
#print(rating2) 

# print(ratings1)
# print(ratings2)
# print(ratings3)
# print(ratings4)


# # /html/body/div[2]/div/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[2]/td[1]