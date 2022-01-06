from bs4.dammit import encoding_res
import pandas as pd
import requests
from lxml import html
from tqdm import tqdm
from bs4 import BeautifulSoup
import csv
import openpyxl 
from openpyxl import Workbook, workbook

url = 'https://asp01.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701'
res = requests.get(url)
data = res.text  
soup = BeautifulSoup(data, "html.parser")
soup.select('#compBody > div.section.ul_de > div:nth-child(3) > div.um_table > table > tbody')
print(soup) 
