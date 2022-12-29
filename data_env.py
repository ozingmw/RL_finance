import pandas as pd
import matplotlib.pyplot as plt
import random

class data_env:
    def __init__(self, path, symbol, max_episodes=250, render_mode=False):
        self.path = path
        self.symbol = symbol
        self.render_mode = render_mode
        self.episodes = 1
        self.max_episodes = max_episodes

        self.df = pd.read_csv(f'{path}/{symbol}.csv')
        self.state_pointer = random.randint(0, len(self.df)-(self.max_episodes+1))
        self.state = self.df.iloc[self.state_pointer]
        self.columns = self.df.columns

        self.balance = 100000000
        self.cash_balance = self.balance
        self.stock_balance = 0
        self.tax1 = 0.004971487
        self.tax2 = 0.001271487
        self.tax3 = 0.001171487
        self.trading_tax = 0.0023
        self.buy_average = 0
        self.buy_counts = 0

        self.time_list = []
        self.done = False

    def _reset(self):
        self.__init__(self.path, self.symbol)
        return self._get_state()

    def step(self, action, counts=0):
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

            ### reward
                before_balance - after_balance
        '''

        if self.done:
            self._reset()
            return self._get_state(), 0, False, self._get_info()

        self.before_balance = self.balance

        # 행동
        # 현재 행동에 대한 결과 및 보상 계산
        # [0, 1, 2] -> ['매수', '매도', '관망']
        if action == 0:
            trading_cost = self.state['Close'] * counts
            if self.cash_balance < trading_cost:
                counts = int(self.cash_balance / self.state['Close'])
                trading_cost = self.state['Close'] * counts
            self.buy_average = (self.buy_average + trading_cost) / (self.buy_counts + counts)
            self.buy_counts += counts
            if trading_cost < 500000:
                self.cash_balance -= int(trading_cost * self.tax1)
            elif trading_cost < 3000000:
                self.cash_balance -= int(trading_cost * self.tax2) + 2000
            elif trading_cost < 30000000:
                self.cash_balance -= int(trading_cost * self.tax2) + 1500
            else:
                self.cash_balance -= int(trading_cost) * self.tax3
            self.cash_balance -= trading_cost
            self.stock_balance += trading_cost

            # 시각화를 위한 저장
            self.time_list.append([0, self.state['Time'], self.state['Close']])

        elif action == 1:
            counts = min(self.buy_counts, counts)
            trading_cost = self.state['Close'] * counts
            if trading_cost < 500000:
                self.cash_balance -= int(trading_cost * self.tax1)
            elif trading_cost < 3000000:
                self.cash_balance -= int(trading_cost * self.tax2) + 2000
            elif trading_cost < 30000000:
                self.cash_balance -= int(trading_cost * self.tax2) + 1500
            else:
                self.cash_balance -= int(trading_cost) * self.tax3
            self.cash_balance -= trading_cost * self.trading_tax
            self.cash_balance += self.state['Close'] * counts
            self.stock_balance -= self.state['Close'] * counts
            self.buy_counts -= counts
            if self.buy_counts == 0:
                self.buy_average = 0

            # 시각화를 위한 저장
            self.time_list.append([1, self.state['Time'], self.state['Close']])

        self.stock_balance = self.state['Close'] * self.buy_counts
        self.balance = self.cash_balance + self.stock_balance
        reward = self.balance - self.before_balance

        if self.render_mode:
            self.render()

        # 모든 행동 끝나고 다음 스텝 준비
        self.state_pointer += 1
        self.episodes += 1
        
        # episode 제한없이 처음부터 데이터끝까지 학습할 때
        # if self.state_pointer >= len(self.df):
        #     self.done = True
        
        # epsode 제한하여 최대 episode만큼만 학습할 때
        if self.episodes >= self.max_episodes:
            self.done = True

        self.state = self.df.iloc[self.state_pointer]

        return self._get_state(), reward, self.done, self._get_info()

    def _get_state(self):
        return self.df.iloc[self.state_pointer:self.state_pointer+1]
    
    def _get_info(self):
        return {'balance': self.balance, 'stock_balance': self.stock_balance, 'cash_balance': self.cash_balance, 'buy_average': self.buy_average}

    def _change_render(self):
        if self.render_mode:
            self.render_mode = False
        else:
            self.render_mode = True

    def render(self):
        plt.figure(figsize=(12,6))
        train_data = self.df.iloc[:self.state_pointer+1]
        x1 = [str(x) for x in train_data['Time']]
        plt.plot(x1, train_data['Close'].values)
        for time in self.time_list:
            plt.annotate(f'{"BUY" if time[0] == 0 else "SELL"}', xy=(str(time[1]), time[2]), xytext=(str(time[1]), time[2]+300),
                         fontsize=14, arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=10))
            if time[0]:
                plt.scatter(str(time[1]), time[2], c='r')
            else:
                plt.scatter(str(time[1]), time[2], c='b')
        plt.xticks(rotation=15)
        plt.show()