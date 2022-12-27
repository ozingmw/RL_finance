import pandas as pd

class data_env:
    def __init__(self, path, symbol):
        self.path = path
        self.symbol = symbol
        self.df = pd.read_csv(f'{path}/{symbol}.csv')
        self.state_pointer = 0
        self.state = self.df.iloc[self.state_pointer]
        self.colums = self.df.columns
        self.done = False

        self.balance = 100000000
        self.cash_balance = self.balance
        self.stock_balance = 0
        self.tax1 = 0.004971487
        self.tax2 = 0.001271487
        self.tax3 = 0.001171487
        self.trading_tax = 0.0023
        self.buy_average = 0
        self.buy_counts = 0

    def _reset(self):
        self.__init__(self.path, self.symbol)
        return self._get_state()

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
        if self.done:
            self._reset()
            return self._get_state(), 0, False, self._get_info()

        self.before_balance = self.balance

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

        self.stock_balance = self.state['Close'] * self.buy_counts
        self.balance = self.cash_balance + self.stock_balance
        reward = self.balance - self.before_balance

        self.state_pointer += 1
        if self.state_pointer >= len(self.df):
            self.done = True
        else:
            self.state = self.df.iloc[self.state_pointer]

        return self._get_state(), reward, self.done, self._get_info()

    def _get_state(self):
        return self.df.iloc[self.state_pointer:self.state_pointer+1]
    
    def _get_info(self):
        return {'balance': self.balance, 'stock_balance': self.stock_balance, 'cash_balance': self.cash_balance, 'buy_average': self.buy_average}