# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

####### START CODING HERE #######

# construct the intelligent agent.
agent = SarsaAgent(all_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995) 

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # choose initial action (SARSA is on-policy and needs a and a')
    a = agent.choose_action(s)
    # agent interacts with the environment, and collects experience
    for iter in range(500):
        # take action a, observe r, s'
        s_, r, isdone, info = env.step(a)
        # env.render()
        episode_reward += r
        if isdone:
            # terminal update: target = r
            agent.learn(s, a, r, s_, a_next=None, done=True)
            break
        # choose next action a' with current policy
        a_ = agent.choose_action(s_)
        # SARSA update with (s,a,r,s',a')
        agent.learn(s, a, r, s_, a_next=a_, done=False)
        # update current state and action to next step
        s, a = s_, a_
    # Per-episode epsilon decay
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
print('\ntraining over\n')   

# close the render window after training.
env.close()

####### END CODING HERE #######


