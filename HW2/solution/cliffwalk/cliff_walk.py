# -*- coding:utf-8 -*-

import math, os, time, sys
import numpy as np
from random import random, choice
import gym
# from gym_gridworld import CliffWalk
from agent import SarsaAgent, QLearningAgent

# construct the environment
env = gym.make('CliffWalking-v0') 
# size of observation space
num_obs = env.observation_space.n
all_obs = np.arange(num_obs)
# size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# construct the intelligent agent.
# agent = SarsaAgent(all_actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.4)
agent = QLearningAgent(all_actions=all_actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.3)

# start training
for episode in range(1000):
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. 
    # You can comment all render() to turn off the GUI in training process to accelerate your code.
    if (episode+1) % 100 == 0:
        env.render()
    # agent interacts with the environment
    for iter in range(500):
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        if (episode+1) % 100 == 0:
            time.sleep(0.1)
            env.render()
        episode_reward += r
        # print(f"{s} {a} {s_} {r} {isdone} {info}")
        agent.learn(s, a, r, s_, isdone)
        s = s_
        if isdone:
            # time.sleep(0.1)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
print('\ntraining over\n')   
print(agent.q_table)

# close the render window after training.
env.close()

