from bs4.dammit import encoding_res
import pandas as pd
import requests
from lxml import html
from tqdm import tqdm
from bs4 import BeautifulSoup
import csv
import openpyxl 

codes = '091970', '091970', '096640'    
price = []
for code in codes:  
    url = f'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701'
    res = requests.get(url)
    data = res.text  
    soup = BeautifulSoup(data, "html.parser") 
    for i in range(1,24):      
        price.append(soup.select_one("#p_grid1_"+ str(i)))
            
print(price)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
