# encoding=utf-8

"""
Reinforcement Learning: all algothrms in one file. Convenient to summary RL.

author: ninglee
"""

import numpy as np
import pandas as pd

class RL(object):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):

        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_actions = self.table.ix[observation,:]
            state_actions.reindex(np.random.permutation(state_actions.index))
            return state_actions.argmax()
        else:
            return np.random.choice(self.actions)

    def check_state_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append({
                pd.Series(
                    data=[0] * len(self.actions),
                    index=self.actions,
                    name=state
                )
            })

    def learn(self, *kwargs):
        pass


# off-policy
class QLearningTable(RL):
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
    
    def learn(self, s, a, r, s_):

        self.check_state_exist(s_)

        q_predict = self.table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.table.ix[s_,:].max()
        else:
            q_target = r

        # Update/Learn
        self.table.ix[s,a] += self.lr * (q_target - q_predict)
    
    
# on-policy
class SarsaTable(RL):
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions,learning_rate, reward_decay, e_greedy)
        
    def learn(self, s, a, r, s_, a_):
        """
        Sarsa是已经知道了下一步要采取的action，而且他也肯定会采取这个action。所以他的学习是直接基于下一次的action的。
        """
        self.check_state_exist(s_)

        q_predict = self.table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.table.ix[s_,a_] # Sarsa使用下一次采取的action来更新
        else:
            q_target = r # next state is terminal
        self.table.ix[s, a] += self.lr * (q_target - q_predict) # update



