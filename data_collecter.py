import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import time
import random
import datetime

def naver_crawler(symbol, day) -> pd.DataFrame:
    page = 1
    df = pd.DataFrame(columns=["Date", "Price", "diffPrice", "sellPrice", "buyPrice", "volume", "change"])

    for page in range(1, 41):
        URL = f"https://finance.naver.com/item/sise_time.naver?code={symbol}&thistime={day}160000&page={page}"
        headers = {'User-agent': 'Mozilla/5.0'}
        res = requests.get(url=URL, headers=headers)

        html = bs(res.text, 'html.parser')

        trList = html.find_all("tr", {"onmouseover":"mouseOver(this)"})
        for tr in trList:
            tdList = tr.find_all('td')

            execution_time = tdList[0].text.strip()  # 체결시각
            execute_price = int(tdList[1].text.strip().replace(',', ''))  # 체결가
            diff_price = int(tdList[2].text.strip().replace(',', ''))  # 전일비
            sell_price = int(tdList[3].text.strip().replace(',', ''))  # 매도
            buy_price = int(tdList[4].text.strip().replace(',', ''))  # 매수
            volume = int(tdList[5].text.strip().replace(',', ''))  # 거래량
            change = int(tdList[6].text.strip().replace(',', ''))  # 변동량
            
            df.loc[len(df)] = [execution_time, execute_price, diff_price, sell_price, buy_price, volume, change]

        page += 1
        time.sleep(random.uniform(0.01, 0.1))
    
    reversed_df = df.loc[::-1].reset_index(drop=True)
    
    return reversed_df

def to_csv(symbol, start_day, end_day):
    start_day_dt = datetime.datetime.strptime(start_day, '%Y%m%d')
    end_day_dt = datetime.datetime.strptime(end_day, '%Y%m%d')
    search_day = start_day_dt
    while (end_day_dt-search_day).total_seconds() >= 0:
        print(f"크롤링 하는 날짜: {search_day}")
        search_day_str = "".join(search_day.date().isoformat().split("-"))
        naver_crawler(symbol, search_day_str)
        search_day += datetime.timedelta(days=1)

to_csv("005930", "20221220", "20221221")