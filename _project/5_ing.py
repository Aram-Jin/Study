
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate


codes = 'A010580','A091970'

for code in codes: 
    url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode={code}' 
    # print(url)

    res = requests.get(url)
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')
    df = pd.read_html(str(table))
    print(tabulate(df[0], headers='keys', tablefmt='psql'))

# dataframe = fs_data(code)