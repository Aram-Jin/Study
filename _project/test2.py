import requests as re
from bs4 import BeautifulSoup
from pandas import DataFrame, Series
import numpy as np
import pandas as pd


#원하시는 종목의 코드명과 기업명을 아래의 형식으로 적으시면 됩니다.

tickers = {'010580':'지코',
           '091970':'나노캠텍',
          '096640':'멜파스',
          '090740':'연이비앤티',
          '013000':'세우글로벌',
          '131100':'스카이이앤엠',
          '015540':'쎌마테라퓨틱스',
          }


# 매출액 단위는 억원입니다.

# 아래 주소를 본인의 PC에 맞게 설정해주시면 됩니다.
my_folder = 'D:\Study\project'


def get_fnguide_table(tickers):

    for ticker in tickers.keys():
        ''' 경로 탐색'''
        url = re.get("https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701")
        url = url.content

        html = BeautifulSoup(url,'html.parser')
        body = html.find('body')

        fn_body = body.find('div',{'class':'fng_wrap'})
        fn_body2 = fn_body.find('div',{'id':'compBody'})
        fn_body3 = fn_body2.find('div',{'class':'section ul_de'})
        fn_body4 = fn_body3.find('div',{'class':'ul_col2wrap pd_t25'})
        ur_table = fn_body4.find('div',{'class':'um_table'})
        table = ur_table.find('div',{'class':'us_table_ty1 h_fix zigbg_no'})
        tbody = table.find('tbody')
        tr = tbody.find_all('tr')

        Table = DataFrame()
        print(Table.head)


#         for i in tr:
        
#             ''' 항목 가져오기'''
#             category = i.find('tr',{'id':'p_grid1_'+i})
            
#             if category == None:
#                 category = i.find('td')   
        
#             category = category.text.strip()

        
#             '''값 가져오기'''
# #             value_list =[]

#             j = i.find_all('td',{'class':'r'})
            
#             for value in j:
#                 temp = value.text.replace(',','').strip()
                    
#                 try:
#                     temp = float(temp)
#                     value_list.append(temp)
#                 except:
#                     value_list.append(0)
            
#             Table['%s'%(category)] = value_list
            
#             ''' 기간 가져오기 '''    
            
#         #     thead = table.find('thead')
#         #     tr_2 = thead.find('tr',{'class':'td_gapcolor2'}).find_all('th')
                    
#         #     year_list = []
            
#         #     for i in tr_2:
#         #         try:
#         #             temp_year = i.find('span',{'class':'txt_acd'}).text
#         #         except:
#         #             temp_year = i.text
                
#         #         year_list.append(temp_year)
                    
#         #     Table.index = year_list
     
#         # Table = Table.T
                 
# #         '''CSV 파일로 저장'''
    
# #         Table.to_csv('%s/%s.csv'%(my_folder,tickers[code]))
                                
# #     return 

# # # get_fnguide_table(tickers)

