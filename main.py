'''
변경할점
    데이터를 이것저것 받는 대신 %로 스케일을 일정하게함
    따라서 리워드도 %의 차이로 해서 리워드 스케일도 되게함

    data_collecter로 데이터 받는거 변경 및 스케일 조정
    data_env로 최대한 dataframe으로 변환 및 데이터 그대로 함
    agent에서 받은 데이터를 가공해서 함

    action도 매수 매도만 설정
'''

import sys

import time_check
import stock_env
import data_collecter
from data_env import data_env
import agent

SYMBOL = '005930'

stock_env.auth()
account = stock_env.current_account()
balance = int(account['output2'][0]['tot_evlu_amt'])

max_episode = 250
input_days = 1
env = data_env('./data/day', SYMBOL, max_episodes=max_episode, balance=balance)

model = agent.DDPG_agent(SYMBOL, env, time_counts=input_days, balance=balance)
model.load_model()

sys_time = time_check.check()
if sys_time == "w1":
    print("주말(토요일)")
    model.train()
    sys.exit("학습 종료")
elif sys_time == "w2":
    sys.exit("주말(일요일)")
elif sys_time == "a":
    print("장 마감")
    data_collecter.update_all_csv()
    sys.exit("데이터 업데이트 완료")
print("장 중")

while time_check.check() == 'd':

    symbol_price = stock_env.current_price(SYMBOL)
    action = model.predict(symbol_price)

    stock_list = stock_env.current_account()['output1']
    evaluation = stock_env.current_account()['output2'][0]

    if action == 0:
        total_balance = int(evaluation['tot_evlu_amt'])
        available_price = total_balance * value
        available_price = min(evaluation['dnca_tot_amt'], available_price)
        available_counts = int(available_price / symbol_price)
        stock_env.buy(SYMBOL, symbol_price, available_counts)
    elif action == 1:
        for stock in stock_list:
            if stock['pdno'] == SYMBOL:
                break
        counts = stock['hldg_qty']
        stock_env.current_account()['output']
        stock_env.sell(SYMBOL, symbol_price, counts)
    elif action == 2:
        pass


print("장 마감")
data_collecter.update_all_csv()
sys.exit("데이터 업데이트 완료")

'''
단타?
매 초/분 마다 데이터 불러와서
분당 데이터로 학습한 모델에 넣기
-> 오늘 가격중 가장 낮을거같은 가격에 구매
-> 구매 이후 높을거같은 가격에 판매

매 분 마다 데이터 불러와서
일별 데이터로 학습한 모델에 넣기
시장가로 평가된 비율에 따라 전체 잔고중 얼마만큼만 구매
학습한 모델이 판매를 선택하면 전부 판매
'''