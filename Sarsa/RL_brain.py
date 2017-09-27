# encoding=utf-8

"""
Reinforcement Learning: Sarsa

author: ninglee
"""

import numpy as np
import pandas as pd

class SarsaTable(object):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            # 防止两个值相等的action总是选择其中一个，要随机选择
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action



    def learn(self, s, a, r, s_, a_):
        """
        Sarsa是已经知道了下一步要采取的action，而且他也肯定会采取这个action。所以他的学习是直接基于下一次的action的。
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_,a_] # Sarsa使用下一次采取的action来更新
        else:
            q_target = r # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict) # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )


