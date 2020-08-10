#import urllib2
from bs4 import BeautifulSoup
import requests
from datetime import datetime


def isTableDataTagWithValue(tag):
    if(tag.name == 'td'):
        return True
        

def findValues(soup):
    tds = soup.find_all(isTableDataTagWithValue)
    hour_first = tds[0].get_text()
    level_first = tds[1].get_text()
    differnce_first = tds[2].get_text()
    return hour_first, level_first, differnce_first

def findFirstWaterLevel():
    
    today = datetime.now()

    today_formated = today.strftime("%Y-%m-%d")
    today_formated_with_percent = today_formated.replace("-", "%F")


    request = 'https://defesacivil.riodosul.sc.gov.br/index.php?r=externo%2Fmetragem-sensores&data_inicial-w0-disp='
    request += today_formated_with_percent
    request += '&data_inicial='
    request += today_formated
    request += '&data_final-w1-disp='
    request += today_formated_with_percent
    request += '&data_final='
    request += today_formated
    request += '&intervalo=5&_pjax=%23kv-pjax-container-metragem-sensores'

    with requests.session() as session:
        resp = session.get(request)
        if(resp.status_code == 200):

            soup = BeautifulSoup(resp.text, "html.parser")

            return findValues(soup)


    