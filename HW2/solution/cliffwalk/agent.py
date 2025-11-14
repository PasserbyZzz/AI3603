# -*- coding:utf-8 -*-
import math, os, time, sys
import pdb
import numpy as np
import gym
import pandas as pd


class QLearningAgent(object):
    def __init__(self, all_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.3, epsilon_decay_rate=0.999):
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay_rate = epsilon_decay_rate
        self.q_table = pd.DataFrame({},
                                    columns=self.all_actions,
                                    dtype=np.float32)

    def check_state_exist(self, state):
        """检查状态是否存在，不存在则添加"""
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.all_actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        return True

    def choose_action(self, observation, epsilon_decay=True):
        """ epsilon greedy"""
        self.check_state_exist(observation)
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # import pdb; pdb.set_trace()
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.all_actions)
        if epsilon_decay:
            self.epsilon *= self.epsilon_decay_rate
        return action
    
    def learn(self, s, a, r, s_, isdone):
        """通过经验学习"""
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if isdone == False:
            self.q_table.loc[s_, :].max()
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

class DynaQAgent(object):
    def __init__(self, all_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.3, epsilon_decay_rate=0.999):
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay_rate = epsilon_decay_rate
        self.q_table = pd.DataFrame({},
                                    columns=self.all_actions,
                                    dtype=np.float32)
        self.database = pd.DataFrame({}, columns=self.all_actions, dtype=np.object)

    def check_state_exist(self, state):
        """检查状态是否存在，不存在则添加"""
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.all_actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        return True

    def choose_action(self, observation, epsilon_decay=True):
        """ epsilon greedy"""
        self.check_state_exist(observation)
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.all_actions)
        if epsilon_decay:
            self.epsilon *= self.epsilon_decay_rate
        return action
    
    def learn(self, s, a, r, s_, isdone):
        """通过经验学习"""
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if isdone == False:
            self.q_table.loc[s_, :].max()
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    
    def store_transition(self, s, a, r, s_):
        """储存transition"""
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.all_actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.loc[s,a] = (r, s_)

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.loc[s].dropna().index)
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.loc[s, a]
        return r, s_

class SarsaAgent(object):
    def __init__(self, all_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.3, epsilon_decay_rate=0.999):
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay_rate = epsilon_decay_rate
        self.q_table = pd.DataFrame({}, 
                                    columns=self.all_actions,
                                    dtype=np.float32)

    def check_state_exist(self, state):
        """检查状态是否存在，不存在则添加"""
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.all_actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        return True

    def choose_action(self, observation, epsilon_decay=True):
        """ epsilon greedy"""
        self.check_state_exist(observation)
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.all_actions)
        if epsilon_decay:
            self.epsilon *= self.epsilon_decay_rate
        return action
    
    def learn(self, s, a, r, s_, isdone):
        """通过经验学习"""
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if isdone == False:
            a_ = self.choose_action(s_, epsilon_decay=False)
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


class QLearningAgent_numpy(object):
    def __init__(self, all_obs, all_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.3, epsilon_decay_rate=0.999):
        self.all_obs = all_obs
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay_rate = epsilon_decay_rate
        self.q_table = np.zeros((all_obs.size, all_actions.size), dtype=float)

    def choose_action(self,observation, epsilon_decay=True):
        """ epsilon greedy"""
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[observation, :]
            action = state_action.argmax() 
        else:
            action = np.random.choice(self.all_actions)
        if epsilon_decay:
            self.epsilon *= self.epsilon_decay_rate
        return action
    
    def learn(self, s, a, r, s_, isdone):
        q_predict = self.q_table[s, a]
        if isdone == False:
            q_target = r + self.gamma * self.q_table[s_, :].max()  # next state is not terminal
        else:
            q_target = r
        self.q_table[s, a] += self.lr * (q_target - q_predict)  # update


