import pandas as pd from pandas
import DataFrame from urllib.request
import urlopen, Request

def get_html_fnguide(code, gb):
    url=[]
    
    url.append("https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode={code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701")

    if gb>3 :
        return None
    url = url[gb]
    try:
        
        
