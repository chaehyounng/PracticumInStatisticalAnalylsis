# crawling.py

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from time import sleep
from datetime import datetime

def naver_news_crawler(query, s_date, e_date, max_count=100):
    """네이버 뉴스에서 특정 키워드와 날짜 범위에 맞는 기사 100개 크롤링"""
    
    title_text = []
    date_text = []
    link_text = []

    page = 1
    while len(title_text) < max_count:
        url = f"https://search.naver.com/search.naver?where=news&query={query}&sort=1&ds={s_date}&de={e_date}&nso=so:r,p:from{s_date.replace('.','')}to{e_date.replace('.','')},a:&start={page}"
        
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"})
        soup = BeautifulSoup(response.text, 'html.parser')

        atags = soup.select('.news_tit')  # 뉴스 제목
        dates = soup.select('.info_group > span.info')  # 날짜
        links = [atag['href'] for atag in atags]  # 기사 링크

        for atag, date in zip(atags, dates):
            if len(title_text) >= max_count:
                break
            title_text.append(atag.text)
            date_text.append(date.text)
            link_text.append(atag['href'])

        page += 10  # 다음 페이지 이동
        sleep(0.5)  # 네이버 차단 방지

    # 크롤링 결과 데이터프레임 변환
    df = pd.DataFrame({"날짜": date_text, "제목": title_text, "링크": link_text})
    return df