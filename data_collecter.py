import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import datetime
from tqdm import tqdm
import os


def naver_min_crawler(symbol, day) -> pd.DataFrame:
    page = 1
    df = pd.DataFrame(columns=["Time", "Price", "Diff_price", "Sell_price", "Buy_price", "Volume", "Change"])

    for page in tqdm(range(1, 41), desc=f'{symbol}_min'):
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

def naver_day_crawler(symbol, start_day, end_day):
    if not end_day:
        start_day = end_day = datetime.date.today().strftime("%Y%m%d")

    URL = f"https://fchart.stock.naver.com/sise.nhn?symbol={symbol}&timeframe=day&count=10000&requestType=0"
    headers = {'User-agent': 'Mozilla/5.0'}
    res = requests.get(url=URL, headers=headers)
    xml = bs(res.text, 'html.parser')
    
    df = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

    line_list = xml.find_all('item')
    for line in tqdm(line_list, desc=f'{symbol}_day'):
        data_list = line.attrs['data'].split("|")
        if int(start_day) > int(data_list[0]):
            continue

        time = data_list[0]
        open_price = data_list[1]
        high_price = data_list[2]
        low_price = data_list[3]
        close_price = data_list[4]
        volume = data_list[5]

        df.loc[len(df)] = [time, open_price, high_price, low_price, close_price, volume]

        if int(end_day) <= int(data_list[0]):
            break

    # if f'{symbol}.csv' in os.listdir('./data/day'):
    #     df_all = pd.read_csv(f'./data/day/{symbol}.csv')
    #     df = pd.concat([df_all, df])
    
    df.to_csv(f"./data/day/{symbol}.csv", index=False)
    
def _make_update_csv(mu, symbol, start_day=None, end_day=None):
    if not start_day:
        start_day = end_day = datetime.date.today().strftime("%Y%m%d")

    if mu == 'make':
        df_all = pd.DataFrame(columns=["Time", "Price", "Diff_price", "Sell_price", "Buy_price", "Volume", "Change"])
    else:
        df_all = pd.read_csv(f'./data/min/{symbol}.csv')

    start_day_dt = datetime.datetime.strptime(start_day, '%Y%m%d')
    end_day_dt = datetime.datetime.strptime(end_day, '%Y%m%d')
    search_day = start_day_dt
    while (end_day_dt-search_day).total_seconds() >= 0:
        search_day_str = "".join(search_day.date().isoformat().split("-"))
        df = naver_min_crawler(symbol, search_day_str)
        df_all = pd.concat([df_all, df])
        search_day += datetime.timedelta(days=1)

    df_all.to_csv(f"./data/min/{symbol}.csv", index=False)


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

def update_all_csv():
    data_list = os.listdir('./data/min')
    for data in data_list:
        symbol = data.split(".")
        update_csv(symbol[0])
    # data_list = os.listdir('./data/day')
    # for data in data_list:
    #     symbol = data.split(".")
    #     naver_day_crawler(symbol[0])