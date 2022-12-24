import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, BatchNormalization, Dense, Concatenate

class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.L1 = LSTM(256, dropout=0.1, return_sequences=True, kernel_initializer='random_normal') 
        self.B1 = BatchNormalization()
        self.L2 = LSTM(128, dropout=0.1, return_sequences=True, kernel_initializer='random_normal')
        self.B2 = BatchNormalization()
        self.L3 = LSTM(64, dropout=0.1, return_sequences=True, kernel_initializer='random_normal')
        self.B3 = BatchNormalization()
        self.L4 = LSTM(32, dropout=0.1, kernel_initializer='random_normal')
        self.B4 = BatchNormalization()

    def call(self, state):
        x = self.L1(state)
        x = self.B1(x)
        x = self.L2(x)
        x = self.B2(x)
        x = self.L3(x)
        x = self.B3(x)
        x = self.L4(x)
        x = self.B4(x)

        return x

