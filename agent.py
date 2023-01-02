from collections import deque
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Concatenate, Input, GlobalAveragePooling1D
from keras.optimizers import Adam

from transformer_rm_mask import Encoder

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
        self.T = Encoder(num_layers=3, d_model=feature, num_heads=5, d_ff=100, dropout_rate=0.3)
        self.G = GlobalAveragePooling1D()
        self.D = Dense(action_dim, activation='tanh')

    def call(self, state):
        x, _ = self.T(state)
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

class DDPG_agent:
    def __init__(self, symbol, env, time_counts=5, balance=100000000):
        self.DISCOUNT_FACTOR = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.TAU = 0.001

        self.symbol = symbol
        self.env = env
        self.TIME_COUNTS = time_counts
        self.balance = balance

        self.action_dim = 3     #[0, 1, 2] -> ['매수', '매도', '관망']
        self.env_state = self.env.df[:self.env.state_pointer+1]
        self.feature = len(self.env.columns)-1      # 특성 개수 -> open, close, high, low, volumn 5개 (Time 제외)

        self.actor_input_shape = (None, self.TIME_COUNTS, self.feature)
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.actor = Actor(action_dim=self.action_dim, feature=self.feature)
        self.target_actor = Actor(action_dim=self.action_dim, feature=self.feature)
        self.critic = Critic()
        self.target_critic = Critic()
        
        self.actor.build(input_shape=self.actor_input_shape)
        self.target_actor.build(input_shape=self.actor_input_shape)
        state_in = Input((self.TIME_COUNTS, self.feature, ))
        action_in = Input((1, self.action_dim, ))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        # self.actor.summary()
        # self.critic.summary()
        
        self.actor_optimizer = Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.buffer = deque(maxlen=self.BUFFER_SIZE)

        self.save_episode_reward = []

        self.action_kor = {
            0: '매수',
            1: '매도',
            2: '관망'
        }

    def set_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(u'%(asctime)s %(message)s')
        file_handler = logging.FileHandler(f'./logs/{datetime.date.today()}_{self.symbol}.log', encoding='utf-8')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.debug('DEBUG LOGGING')
        return logger

    def update_target_network(self, TAU):
        weights = self.actor.get_weights()
        target_weights = self.target_actor.get_weights()
        for index in range(len(weights)):
            target_weights[index] = TAU * weights[index] + (1-TAU) * target_weights[index]
        self.target_actor.set_weights(target_weights)

        weights = self.critic.get_weights()
        target_weights = self.target_critic.get_weights()
        for index in range(len(weights)):
            target_weights[index] = TAU * weights[index] + (1-TAU) * target_weights[index]
        self.target_critic.set_weights(target_weights)

    def ou_noise(self, x, rho=0, mu=0, dt=1e-1, sigma=0.2, loc=0, scale=1, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(loc=loc, scale=scale, size=dim)

    def td_target(self, rewards, v_values, dones):
        td = np.asarray(v_values)
        for index in range(v_values.shape[0]):
            td[index] = rewards[index] + (1-dones[index]) * self.DISCOUNT_FACTOR * v_values[index]
        return td
        
    def unpack_batch(self, replay_memory):
        indices = np.random.randint(len(replay_memory), size=self.BATCH_SIZE)
        batch = [replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array(
                [experience[field_index] for experience in batch]
            ) for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            states = tf.reshape(states, [self.BATCH_SIZE, -1, 5])
            actions = self.actor(states, training=True)
            critic = self.critic([states, tf.reshape(actions, [self.BATCH_SIZE, -1, 3])])
            loss = -tf.reduce_mean(critic)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            v = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(v - td_targets))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def get_action(self, action):
        action = np.argmax(action, axis=1)
        return action

    def load_model(self):
        self.actor.load_weights(f'./model/{self.symbol}_DDPG_actor.h5')
        self.critic.load_weights(f'./model/{self.symbol}_DDPG_critic.h5')

    def save_model(self):
        self.actor.save_weights(f"./model/train/{self.symbol}_DDPG_actor.h5")
        self.critic.save_weights(f"./model/train/{self.symbol}_DDPG_critic.h5")

    def train(self, max_episode=500):
        # actor [None, 250, 5] -> time_series_data, feature
        # critic [(5,), (3,)] -> feature(current_data), action
        total_reward = 0

        logger = self.set_logger()
        self.update_target_network(1.0)
        
        for episode in range(max_episode):
            pre_noise = np.zeros(self.action_dim)
            episode_step, episode_reward, done = 0, 0, False
            state = self.env.reset()
            state = state.values.tolist()[0][1:]

            # state [batch_size, 1, 5] 에서 [batch_size, time_counts, 5]로 변환중
            while not done:
                state_list = self.env.stack_step(self.TIME_COUNTS)
                state_list = state_list.drop(['Time'], axis=1).values.tolist()

                action = self.actor(tf.convert_to_tensor([state_list], dtype=tf.float32))
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + [noise], -1, 1)
                real_action = self.get_action(action)

                value = self.critic([tf.convert_to_tensor([state_list], tf.float32), action]).numpy()
                
                next_state, reward, done, info = self.env.step(real_action, 1)
                next_state = next_state.values.tolist()[0][1:]
                
                # train_reward = (next_state[3] - state[3]) / state[3] * 100

                # 굳이 clipping 해야하나?
                # 차라리 value clipping 해야하는거 아닌가?
                # value output으로 주식 매매 비율 정하는건데
                # 현재 activation 함수가 linear 이면 너무 높거나 낮고
                # softmax로 변경하면 1로 고정?
                # value output이 초기에 너무 높음 -> 후반에는 몰?루
                # 매매할때 value output으로 하면 안될거같음
                # 그럼 뭘로 대체?
                self.buffer.append((state_list, action, reward, next_state, done))

                if len(self.buffer) > 10:
                    states, actions, rewards, next_states, dones = self.unpack_batch(self.buffer)
                    target_qs = self.target_critic([
                        tf.convert_to_tensor(next_states, dtype=tf.float32),
                        self.target_actor(tf.reshape(tf.cast(next_states, dtype=tf.float32), [self.BATCH_SIZE, -1, 5]))
                    ])

                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.update_target_network(self.TAU)

                pre_noise = noise
                state = next_state
                episode_reward += reward
                episode_step += 1
                
                print(f'Episode: {episode+1}, Episode_step: {episode_step}, '
                     +f'Reward: {episode_reward:.3f}, Action: {self.action_kor[real_action[0]]}\r', end="")
                logger.debug(f'EPISODE: {episode+1}, EPISODE_STEP: {episode_step}, STATE: {state[3]}, NEXT_STATE: {next_state[3]}, '
                                 +f'ACTION: {self.action_kor[real_action[0]]}, REWRD: {reward}, EPISODE_REWARD: {episode_reward}')

            total_reward += episode_reward
            print(f'Episode: {episode+1}, Episode_step: {episode_step}, '
                 +f'Reward: {episode_reward:.3f}, Action: {self.action_kor[real_action[0]]}, Total: {total_reward:.3f}')
            logger.debug(f'EPISODE: {episode+1}, EPISODE_STEP: {episode_step}, STATE: {state[3]}, NEXT_STATE: {next_state[3]}, '
                             +f'ACTION: {self.action_kor[real_action[0]]}, REWRD: {reward}, EPISODE_REWARD: {episode_reward}')
            self.save_episode_reward.append(episode_reward)

            if episode % 10 == 0:
                self.save_model()

    def validation(self):
        state = self.env.reset(training=False)
        state = state.values.tolist()[0][1:]

        episode_reward, done = 0, False
        print("Validation:")
        while not done:
            action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
            real_action = self.get_action(action)
            state, reward, done, info = self.env.step(real_action, 1)
            state = state.values.tolist()[0][1:]
            episode_reward += reward
            print(f'Reward: {episode_reward:.3f}, Action: {self.action_kor[real_action[0]]}')
        self.env.render()

    def predict(self, env):
        state = env.reset(training=False)
        state = state.values.tolist()[0][1:]
        action = self.actor(tf.convert_to_tensor(state, dtype=tf.float32))
        action = self.get_action(action)
        return action

    def plot_result(self):
        plt.plot(self.save_episode_reward)
        plt.show()


from data_env import data_env
symbol = '052400'
max_episodes_step = 250
max_episodes = 500
input_days = 5
balance = 100000000
env = data_env('./data/day', symbol, max_episodes_step=max_episodes_step, balance=balance)
agent = DDPG_agent(symbol, env, time_counts=input_days, balance=balance)
agent.train(max_episode=max_episodes)
agent.plot_result()