'''
main
    현재 시간 확인
    9시 이전이면 9시 까지 대기

    장 시작 후
        현재 가격 확인
        모델 판단
        매매 결정
        거래

    장 마감 후
        이전 데이터에 오늘 데이터 추가
        새로운 데이터로 재 학습

# stock_env
#     인증
#     현재 잔고 확인
#     현재 가격 확인
#     매수
#     매도
#     오늘 거래내역 확인

agent_env
    매수, 매도, 관망
    

agent
    transformer + SAC모델
    시계열 데이터를 transformer에 넣어 ???
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
    
'''

import time_check
import stock_env
# import agent

time_check.check()


SYMBOL = '005930'

# env.auth()
stock_env.current_account()
symbol_current_price = stock_env.current_price(symbol=SYMBOL)