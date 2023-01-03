'''
main
    # 현재 시간 확인
    # 9시 이전이면 9시 까지 대기
    # 15시 이후라면 오늘 데이터 추가 및 모델 추가 학습

    장 시작 후
        현재 가격 확인
        모델 판단
        매매 결정
        거래

    장 마감 후
        이전 데이터에 오늘 데이터 추가
        새로운 데이터로 재 학습

    매 프로그램마다 각자 SYMBOL에 맞는 것만 판단
    따라서 여러 프로그램이 실행되어야함
    -> 이 때 학습을 진행하면 여러 프로그램 학습 안되는거 아녀?

stock_env
    # 인증
    # 현재 잔고 확인
    # 현재 가격 확인
    # 매수
    # 매도
    # 오늘 거래내역 확인

data_env
    매수, 매도, 관망
    

agent
    GRU + A2C모델
    시계열 데이터를 GRU에 넣어 ???
    actor:
        input: 이전가격, 현재가격, ???
        output: 1.매수, 2.매도, 3.관망
    critic: 
        input: 현재가격, 행동, ???
        output: 행동의 평가 지표 -> 정규화하여 (0, 1) 일부분만 매매
    
    주식 가격 예측(종가)
    예측된 종가가 현재가보다 낮을때
        오늘 주식 흐름 예측하여 저점 예측 후 지정가 매수
        다음날 종가 예측 후 다음날도 상승이면 매수 이후 관망
    예측된 종가가 현재가보다 높을때
        흐름 예측하여 고점 예측 후 지정가 매도
    
현재 data_env에서 종가로 비교하는데
reward normalization 때문에 가격 그대로보단 산 가격하고 다음 스텝 종가하고 등락률로 변경해서 넣는게 좋아보임
-> 좋아보여서 수정중
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