import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

code = '131100'
url = f'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{code}'
# print(url)
res = requests.get(url)
df = pd.read_html(res.text)
print(tabulate(df[0], headers='keys', tablefmt='psql'))

