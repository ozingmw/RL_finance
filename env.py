import requests
import json

with open('config.json', 'r') as f:
    _config = json.load(f)
APP_KEY = _config['APP_KEY']
APP_SECRET = _config['APP_SECRET']
CANO = _config["CANO"]
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6IjUxNGE3NmNiLWVhNmQtNGZiZi04MjhmLTA4Y2I5MmIzMDgzNSIsImlzcyI6InVub2d3IiwiZXhwIjoxNjcxNDQ2NjAyLCJpYXQiOjE2NzEzNjAyMDIsImp0aSI6IlBTU29UeXVaMngzelpUbmh4OXpGUTNvTlZaZUhORUpuYUdVaCJ9.-y9maV05NuZUIUunPVaGoY33bOMlTLjh6R01GCG_3GWAPIacVZ8r5oIzxrFcGyE-ba6vlMI2mqvuN4Cxf6BKGg"
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
    return res

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