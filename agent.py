from collections import deque
import numpy as np
from keras.models import Model
from keras.layers import GRU, BatchNormalization, Dense, Concatenate, Input
from keras.optimizers import Adam

class Actor(Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        
        self.L1 = GRU(256, dropout=0.1, return_sequences=True, kernel_initializer='he_normal') 
        self.B1 = BatchNormalization()
        self.L2 = GRU(128, dropout=0.1, return_sequences=True, kernel_initializer='he_normal')
        self.B2 = BatchNormalization()
        self.L3 = GRU(64, dropout=0.1, return_sequences=True, kernel_initializer='he_normal')
        self.B3 = BatchNormalization()
        self.D = Dense(action_dim, activation='sigmoid')

    def call(self, state):
        x = self.L1(state)
        x = self.B1(x)
        x = self.L2(x)
        x = self.B2(x)
        x = self.L3(x)
        x = self.B3(x)
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
    def __init__(self, symbol, env):
        self.symbol = symbol
        self.env = env
        self.action_dim = 3     #[0, 1, 2] -> ['매수', '매도', '관망']
        self.env_state = self.env.df[:self.env.state_pointer+1]
        self.actor_input_shape = (None, len(self.env.columns)-1, len(self.env_state))
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.actor = Actor(action_dim=self.action_dim)
        self.target_actor = Actor(action_dim=self.action_dim)
        self.critic = Critic()
        self.target_critic = Critic()
        self.actor.build(input_shape=self.actor_input_shape)
        self.target_actor.build(input_shape=self.actor_input_shape)
        state_in = Input((len(self.env.columns)-1), )
        action_in = Input((self.action_dim, ))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])
        
        self.actor_optimizer = Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.save_episode_reward = []

    def get_action(self, action_output):
        action = np.argmax(action_output)
        return action

    def load_model(self):
        self.actor.load_weights(f'./model/{self.symbol}_actor.h5')
        self.critic.load_weights(f'./model/{self.symbol}_critic.h5')

    def save_model(self):
        self.actor.save_weights(f"./model/{self.symbol}_actor.h5")
        self.critic.save_weights(f"./model/{self.symbol}_critic.h5")

    def train(self):
        max_episode = 1000
        
        for episode in range(max_episode):
            self.env._reset()
            action, value, total_reward, done = 2, 0, 0, False
            batch_memory = deque(maxlen=self.batch_size)

            while not done:
                next_state, reward, done, info = self.env.step(action, value)
                total_reward += reward
                
                if done:
                    break

    def predict():
        pass

from data_env import data_env
symbol = '005930'
env = data_env('./data/day', symbol, False)
agent = OO_agent(symbol, env)