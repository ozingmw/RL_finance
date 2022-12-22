import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import datetime
from tqdm import tqdm

def naver_crawler(symbol, day) -> pd.DataFrame:
    page = 1
    df = pd.DataFrame(columns=["Time", "Price", "Diff_price", "Sell_price", "Buy_price", "Volume", "Change"])

    for page in tqdm(range(1, 41), desc=day):
        URL = f"https://finance.naver.com/item/sise_time.naver?code={symbol}&thistime={day}160000&page={page}"
        headers = {'User-agent': 'Mozilla/5.0'}
        res = requests.get(url=URL, headers=headers)

        html = bs(res.text, 'html.parser')

        trList = html.find_all("tr", {"onmouseover":"mouseOver(this)"})
        for tr in trList:
            tdList = tr.find_all('td')
            
            if tdList[0].text.strip() == "":
                break   
            execution_time = tdList[0].text.strip()  # 체결시각
            execute_price = int(tdList[1].text.strip().replace(',', ''))  # 체결가
            diff_price = tdList[2].text.strip().replace(',', '')  # 전일비
            sell_price = int(tdList[3].text.strip().replace(',', ''))  # 매도
            buy_price = int(tdList[4].text.strip().replace(',', ''))  # 매수
            volume = int(tdList[5].text.strip().replace(',', ''))  # 거래량
            change = int(tdList[6].text.strip().replace(',', ''))  # 변동량
            
            if tdList[2].select('img'):
                plus_minus = tdList[2].select('img')[0]['alt']
                plus_minus = '-' if plus_minus == '하락' else ''
                diff_price = int(plus_minus + diff_price)
            
            execution_time = day + " " + execution_time

            df.loc[len(df)] = [execution_time, execute_price, diff_price, sell_price, buy_price, volume, change]

        page += 1
    
    reversed_df = df.loc[::-1].reset_index(drop=True)
    
    return reversed_df

def _make_update_csv(mu, symbol, start_day=None, end_day=None):
    if not start_day:
        start_day = end_day = datetime.date.today().strftime("%Y%m%d")

    if mu == 'make':
        df_all = pd.DataFrame(columns=["Time", "Price", "Diff_price", "Sell_price", "Buy_price", "Volume", "Change"])
    else:
        df_all = pd.read_csv(f'./data/{symbol}.csv')

    start_day_dt = datetime.datetime.strptime(start_day, '%Y%m%d')
    end_day_dt = datetime.datetime.strptime(end_day, '%Y%m%d')
    search_day = start_day_dt
    while (end_day_dt-search_day).total_seconds() >= 0:
        search_day_str = "".join(search_day.date().isoformat().split("-"))
        df = naver_crawler(symbol, search_day_str)
        df_all = pd.concat([df_all, df])
        search_day += datetime.timedelta(days=1)

    df_all.to_csv(f"./data/{symbol}.csv", index=False)


def make_csv(symbol, start_day=None, end_day=None):
    '''
    start_day, end_day\n
    default value -> today
    '''
    _make_update_csv('make', symbol, start_day, end_day)

def update_csv(symbol, start_day=None, end_day=None):
    '''
    start_day, end_day\n
    default value -> today
    '''
    _make_update_csv('update', symbol, start_day, end_day)