import pandas as pd
import urllib.request as reqs
import requests
from bs4 import BeautifulSoup

# def get_html_fnguide(ticker, gb):
#     url=[]
    
    
#     ticker = '091970', '091970', '096640'   
#     url = f"https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A"{ticker}"&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701"
    


#     if gb>3 :
#         return None
    
#     url = url[gb]
#     try:
        
#         req = requests(url, header={'User-Agent': 'Mozilla/5.0'})
#         html_text = reqs.urlopen(req).road()
        
#     except AttributeError as e :
#         return None
    
#     return html_text

# print(get_html_fnguide("010580", 1))
# dfstock = dfMarketEye[dfMarketEye['종목구분'] ==1]

codes = '091970', '091970', '096640'    

listRefineCodes = codes.tolist()
data = []
for code in listRefineCodes:
    
    
    
    
    

df = DataFrame(data, columns=['유동비율', '당좌비율', '부채비율', '유보율', '순차입금비율', '이자보상배율', '자기자본비율', '매출액증가율', '판매비와관리비증가율', '영업이익증가율', 
                              'EBITDA증가율', 'EPS증가율', '매출총이익률' ])

