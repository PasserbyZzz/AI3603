# -*- coding:utf-8 -*-
# RL task in cliff-walking environment
import math, os, time, sys
import pdb
import numpy as np
import random
import gym
# from gym_gridworld import CliffWalk
# from agent import AgentExample

# construct the environment
env = gym.make('CliffWalking-v0') 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 


# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = int(input("input an action:"))
        s_, r, isdone, info = env.step(a)
        env.render()
        # update the episode reward
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone} {info}")
        # agent learns from experience
        s = s_
        if isdone:
            time.sleep(0.5)
            break
    print('episode:', episode, 'episode_reward:', episode_reward)  
print('\ntraining over\n')   

# close the render window after training.
env.close()




