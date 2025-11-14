# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import Dyna_QAgent
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

##### START CODING HERE #####

# construct the intelligent agent.
agent = Dyna_QAgent(all_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_planning=20)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment, and collects experience
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        # take action a, observe r, s'
        s_, r, isdone, info = env.step(a)
        # env.render()
        # update the episode reward
        episode_reward += r
        # Dyna-Q learn: one real update + planning updates
        agent.learn(s, a, r, s_, done=isdone)
        # update current state to next state
        s = s_

        # check if episode is done
        if isdone:
            break
    # Per-episode epsilon decay
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
print('\ntraining over\n')   

# close the render window after training.
env.close()

##### END CODING HERE #####


