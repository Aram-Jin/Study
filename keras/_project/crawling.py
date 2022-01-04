# requests, beatifulsoup
from bs4 import BeautifulSoup
from urllib.request import urlopen
# with urlopen('https://en.wikipedia.org/wiki/Main_Page') as response:
#     soup = BeautifulSoup(response, 'html.parser')
#     for anchor in soup.find_all('a'):
#         print(anchor.get('href', '/'))
        
response = urlopen('https://www.naver.com/')
soup = BeautifulSoup(response, 'html.parser')
for anchor in soup.find_all('a'):
    print(anchor.get('href', '/'))
    

        
