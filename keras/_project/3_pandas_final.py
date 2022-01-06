import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

# codes = ['096640','091970']
# for code in codes: 
# url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=B&NewMenuID=104&stkGb=701'
# print(url)

res = requests.get('http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701')
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')
df = pd.read_html(str(table))]v 
print(tabulate(df[0], headers='keys', tablefmt='psql'))

# dataframe = fs_data(code)
