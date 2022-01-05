import pandas as pd
import urllib.request as reqs
import requests
from bs4 import BeautifulSoup

def get_html_fnguide(ticker, gb):
    url=[]
    
    url.append("https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701")
    


    if gb>3 :
        return None
    
    url = url[gb]
    try:
        
        req = requests(url, header={'User-Agent': 'Mozilla/5.0'})
        html_text = reqs.urlopen(req).road()
        
    except AttributeError as e :
        return None
    
    return html_text

print(get_html_fnguide("010580", 1))
# dfstock = dfMarketEye[dfMarketEye['종목구분'] ==1]

