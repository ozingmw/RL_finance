import requests
import json
import datetime

with open('config.json', 'r') as f:
    _config = json.load(f)
APP_KEY = _config['APP_KEY']
APP_SECRET = _config['APP_SECRET']
CANO = _config["CANO"]
ACCESS_TOKEN = ""
URL_BASE = "https://openapivts.koreainvestment.com:29443"

# Auth
def auth():
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"content-type":"application/json"}
    params = {
        "grant_type":"client_credentials",
        "appkey":APP_KEY, 
        "appsecret":APP_SECRET
    }
    res = requests.post(URL, headers=headers, data=json.dumps(params))
    
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
            print(f"{stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주 | "
                 +f"현재가: {stock['prpr']}원 | "
                 +f"오늘 등락: {float(stock['fltt_rt']):.2f}% | "
                 +f"매입단가: {stock['pchs_avg_pric']}원 | "
                 +f"수익률: {stock['evlu_pfls_rt']}% | ")
        # 오류시 complete 참고

    print(f"주식 평가 금액: {evaluation[0]['scts_evlu_amt']}원")
    print(f"평가 손익 합계: {evaluation[0]['evlu_pfls_smtl_amt']}원")
    print(f"총 평가 금액: {evaluation[0]['tot_evlu_amt']}원")
    print(f"=====================")
    return res.json()

def complete(start_time=None, end_time=None):
    '''
    FORMAT: 'YYYYMMDD'
    DEFAULT -> Today
    '''

    if not start_time:
        start_time = end_time = datetime.datetime.today().strftime('%Y%m%d')

    PATH = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type":"application/json", 
        "authorization": f"Bearer {ACCESS_TOKEN}",
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

    for stock in stock_list:
        # 매수    삼성전자    평균 매수가: 60000원, 체결수량: 10주, 총 체결금액: 600000원
        print(f"{'매도' if stock['sll_buy_dvsn_cd'] == '01' else '매수'} | {stock['prdt_name']} | "\
             +f"주문단가: {stock['ord_unpr']} | 주문수량: {stock['ord_qty']} | 총 체결금액: {stock['tot_ccld_amt']}")

    print(f"총 주문수량: {total_list['tot_ord_qty']} | 총 체결수량: {total_list['tot_ccld_qty']} | 총 체결금액:{total_list['tot_ccld_amt']}")

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
        'ORD_DVSN': "00", 
        'ORD_QTY': str(counts),
        'ORD_UNPR': str(price)
    }
    res = requests.post(URL, headers=headers, data=json.dumps(params))
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
    res = requests.post(URL, headers=headers, data=json.dumps(params))
    print(res.json()['msg1'])
    return res.json()