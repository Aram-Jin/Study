from selenium import webdriver

browser = webdriver.Chrome("./chromedriver_win32/chromedriver.exe")
browser.get("http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A091970&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701")