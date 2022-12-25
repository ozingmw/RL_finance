import tensorflow as tf
from keras.models import Model
from keras.layers import GRU, BatchNormalization, Dense, Concatenate

class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.L1 = GRU(256, dropout=0.1, return_sequences=True, kernel_initializer='he_normal') 
        self.B1 = BatchNormalization()
        self.L2 = GRU(128, dropout=0.1, return_sequences=True, kernel_initializer='he_normal')
        self.B2 = BatchNormalization()
        self.L3 = GRU(64, dropout=0.1, return_sequences=True, kernel_initializer='he_normal')
        self.B3 = BatchNormalization()
        self.D = Dense(1)

    def call(self, state):
        x = self.L1(state)
        x = self.B1(x)
        x = self.L2(x)
        x = self.B2(x)
        x = self.L3(x)
        x = self.B3(x)
        x = self.L4(x)
        x = self.B4(x)
        x = self.D(x)

        return x

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.x1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state, action = state_action[0], state_action[1]
        
        x = self.x1(state)
        x_a = Concatenate(axis=-1)([x, action])
        v = self.h2(x_a)
        v = self.h3(v)
        v = self.q(v)

        return v


class OO_agent:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()

    def load_model(self, path):
        pass

    def save_model(self, symbol):
        self.actor.save_weights(f"./model/{symbol}_actor.h5")
        self.critic.save_weights(f"./model/{symbol}_critic.h5")

    def train():
        pass

    def predict():
        pass
