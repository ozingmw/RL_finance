import pandas as pd

class data_env:
    def __init__(self, path, symbol):
        self.df = pd.read_csv(f'{path}/{symbol}.csv')
        self.state_pointer = 0
        self.state = self.df.iloc[self.state_pointer]
        self.colums = self.df.columns

        self.balance = 100000000
        self.tax = 3.3
        self.buy_average = 0
        self.buy_counts = 0

    def step(self, action, counts):
        '''
            ### action
                0: buy
                1: sell
                2: hold
            
            ### obs
                day
                    Time, Open, High, Low, Close, Volume
                min
                    Time, Price, Diff_price, Sell_price, Buy_price, Volume, Change
        '''
        if action == 0:
            self.buy_average = (self.buy_average + self.state['Close'] * counts) / (self.buy_counts + counts)
            self.buy_counts += counts
        elif action == 1:
            self.balance += self.state['Close'] * self.buy_counts
            self.return_rate = (self.buy_average - self.state['close']) / self.buy_average
            self.buy_counts = 0
            self.buy_average = 0

        # reward 구상
        # 팔 때 수익률로 리워드 추가?
        # 그럼 살땐?
        # 세금 계산해서 포함해야함
        # 세금있어서 살때 세금때서 바로 수익률 -일텐데
        # 이렇게 되면 매 스텝마다 수익률 계산해서 구하는것도 괞찬을듯

        self.state_pointer += 1
        self.state = self.df.iloc[self.state_pointer:self.state_pointer+1]
        return self._get_state()

    def _get_state(self):
        return self.state

a = data_env('./data/day', '005930')
a.step()