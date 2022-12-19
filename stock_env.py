import requests
import json
import datetime

with open('config.json', 'r') as f:
    _config = json.load(f)
APP_KEY = _config['APP_KEY']
APP_SECRET = _config['APP_SECRET']
CANO = _config["CANO"]
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6IjVlM2I5ODcxLTM1NDItNDMwYy1hZjAwLTU5NWMxYzJiMGFiYyIsImlzcyI6InVub2d3IiwiZXhwIjoxNjcxNTQzNDg5LCJpYXQiOjE2NzE0NTcwODksImp0aSI6IlBTU29UeXVaMngzelpUbmh4OXpGUTNvTlZaZUhORUpuYUdVaCJ9.0O3NEXmTwjEhS7Rj4NvCd9LgP7r9F7ZEWYJygRwWgy1v3R2Ufntc8xivfRQUUb8kCxwTgEU2nNcS6QLGqlg7Fw"
URL_BASE = "https://openapivts.koreainvestment.com:29443"

# Auth
def auth():
    headers = {"content-type":"application/json"}
    body = {
        "grant_type":"client_credentials",
        "appkey":APP_KEY, 
        "appsecret":APP_SECRET
    }
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    
    global ACCESS_TOKEN
    ACCESS_TOKEN = res.json()["access_token"]
    # print(ACCESS_TOKEN)

def current_account():
    PATH = "uapi/domestic-stock/v1/trading/inquire-balance"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type":"application/json", 
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"VTTC8434R",
        "custtype":"P",
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": "",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(URL, headers=headers, params=params)

    stock_list = res.json()['output1']
    evaluation = res.json()['output2']
    stock_dict = {}

    print(f"====주식 보유잔고====")
    for stock in stock_list:
        if int(stock['hldg_qty']) > 0:
            stock_dict[stock['pdno']] = stock['hldg_qty']
            print(f"{stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주, \
                     매입단가: {stock['pchs_avg_pric']}원, \
                     현재가: {stock['prpr']}, \
                     수익률: {stock['evlu_pfls_rt']}, \
                     전일대비: {stock['bfdy_cprs_icdc']}, \
                     등락: {stock['fltt_rt']}")

    print(f"주식 평가 금액: {evaluation[0]['scts_evlu_amt']}원")
    print(f"평가 손익 합계: {evaluation[0]['evlu_pfls_smtl_amt']}원")
    print(f"총 평가 금액: {evaluation[0]['tot_evlu_amt']}원")
    print(f"=====================")
    return res.json()

def complete(start_time=None, end_time=None):
    '''
    FORMAT: 'YYYYMMDD'
    DEFAULT -> todays
    '''

    if not start_time:
        start_time = end_time = datetime.datetime.today().strftime('%Y%m%d')

    PATH = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type":"application/json", 
        "authorization": ACCESS_TOKEN,
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"VTTC8001R"
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": "",
        "INQR_STRT_DT": str(start_time),
        "INQR_END_DT": str(end_time),
        "SLL_BUY_DVSN_CD": '00',
        "INQR_DVSN": '00',
        "PDNO": "",
        "CCLD_DVSN": "00",
        "ORD_GNO_BRNO": "",
        "ODNO":"",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "INQR_DVSN_2": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(URL, headers=headers, params=params)
    stock_list = res.json()['output1']
    total_list = res.json()['output2']

    # 출력값 정렬

    return res.json()

def current_price(symbol):
    PATH = "uapi/domestic-stock/v1/quotations/inquire-price"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"FHKST01010100"}
    params = {
        "fid_cond_mrkt_div_code":"J",
        "fid_input_iscd": symbol
    }
    res = requests.get(URL, headers=headers, params=params)

    if res.status_code == 200 and res.json()["rt_cd"] == "0" :
        return(res.json()['output']['stck_prpr'])
    # 토큰 만료 시
    elif res.status_code == 200 and res.json()["msg_cd"] == "EGW00123" :
        auth()
        current_price(symbol)
    else:
        print("Error Code : " + str(res.status_code) + " | " + res.text)
        return None

def buy(symbol, price, counts):
    PATH = "/uapi/domestic-stock/v1/trading/order-cash"
    URL = f'{URL_BASE}/{PATH}'
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"VTTC0802U"}
    params = {
        'CANO': str(CANO),
        'ACNT_PRDT_CD': "00",
        'PDNO': str(symbol), 
        'ORD_DVSN': "01", 
        'ORD_QTY': str(counts),
        'ORD_UNPR': str(price)
    }
    res = requests.post(URL, headers=headers, params=params)
    print(res.json()['msg1'])
    return res.json()

def sell(symbol, price, counts):
    PATH = "/uapi/domestic-stock/v1/trading/order-cash"
    URL = f'{URL_BASE}/{PATH}'
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"VTTC0801U"}
    params = {
        'CANO': str(CANO),
        'ACNT_PRDT_CD': "00",
        'PDNO': str(symbol), 
        'ORD_DVSN': "01", 
        'ORD_QTY': str(counts),
        'ORD_UNPR': str(price)
    }
    res = requests.post(URL, headers=headers, params=params)
    print(res.json()['msg1'])
    return res.json()

