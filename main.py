'''
현재 데이터 수집
현재 데이터를 에이전트에 전달
에이전트 판단
판단 결과 다시 메인으로 전달
판단에 따라 매매, 관망 후 결과 보고
수집한 데이터 저장
특정 시간 마다 에이전트 업데이트
'''

realtime_data = data_collecter.get_realtime_data()

trading_state = agent.trading(realtime_data)

trader(trading_state)

data_collecter.update()