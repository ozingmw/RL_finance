from collections import deque
import numpy as np
from keras.models import Model
from keras.layers import Dense, Concatenate, Input, GlobalAveragePooling1D
from keras.optimizers import Adam

from transformer import Encoder

'''
    지금 모델
        actor 모델에 하루 데이터(open,close,high,low) gru에 넣어서 행동 결정
        critic 모델에 하루 데이터와 actor model output 넣어서 행동 가치 결정

    변경할 모델
    1.
        1. 현재 가격 예측
            며칠 데이터(open,close,high,low, ...) gru에 넣어서 다음날(다음 며칠간) 가격 예측
        2. 행동 결정
            actor 모델에 이전 며칠 데이터 + 다음날(다음 며칠간 데이터) 넣어서 행동 결정
            critic 모델에 이전 며칠 데이터 + 다음날(다음 며칠간 데이터) 넣어서 행동 가치 결정

    학습할 때
        데이터 open,close,high,low로 확률 밀도함수로 만들어서
        데이터 변환, env실행할 때마다 변경
        모델 출력은 그대로 다음날 가격 예측, 값 하나
'''

class Actor(Model):
    def __init__(self, action_dim, feature):
        super(Actor, self).__init__()
        
        # d_model = input shape / d_model % num_head == 0 / d_ff = dense units
        self.T = Encoder(num_layers=1, d_model=feature, num_heads=5, d_ff=100, dropout_rate=0.3)
        self.G = GlobalAveragePooling1D()
        self.D = Dense(action_dim, activation='softmax')

    def call(self, state):
        x, _ = self.T(state, None)
        x = self.G(x)
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
    def __init__(self, symbol, env, time_counts):
        self.symbol = symbol
        self.env = env
        self.action_dim = 3     #[0, 1, 2] -> ['매수', '매도', '관망']
        self.env_state = self.env.df[:self.env.state_pointer+1]
        self.TIME_COUNTS = time_counts
        self.feature = len(self.env.columns)-1      # 특성 개수 -> open, close, high, low, volumn 5개
        self.BATCH_SIZE = 32

        self.actor_input_shape = (None, self.TIME_COUNTS, self.feature)
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.actor = Actor(action_dim=self.action_dim, feature=self.feature)
        self.target_actor = Actor(action_dim=self.action_dim, feature=self.feature)
        self.critic = Critic()
        self.target_critic = Critic()
        
        self.actor.build(input_shape=self.actor_input_shape)
        self.target_actor.build(input_shape=self.actor_input_shape)
        state_in = Input((self.feature), )
        action_in = Input((self.action_dim, ))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])
        
        self.actor_optimizer = Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.save_episode_reward = []

    def get_action(self, action_output):
        action = np.argmax(action_output)
        return action

    def unpack_batch(self):
        pass

    def td_target(self):
        pass

    def log_pdf(self):
        pass

    def action_learn(self):
        pass

    def critic_learn(self):
        pass

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
max_episodes = 250
env = data_env('./data/day', symbol, max_episodes=max_episodes)
agent = OO_agent(symbol, env, time_counts=max_episodes)