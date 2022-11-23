import sys
import random
import math
from gym import Env, spaces

from data import *
from agent import SupplyChain

class BeerGameEnv(Env):

    def __init__(self):
        
        # observations space coded inventory levels
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(9, start=1), 
             spaces.Discrete(9, start=1), 
             spaces.Discrete(9, start=1), 
             spaces.Discrete(9, start=1)) 
        )        
        
        # number of actions is 4^4
        self.action_space = spaces.Discrete(256) 


    def custom_init(self, data):
        self.sc = SupplyChain(data)

        # initial state values for all levels in supply chain is 12
        # current state 
        self.state = self.sc.encode_tuple((12,12,12,12))

    def step(self, action):
        assert self.action_space.contains(action)
        obs, reward, done, info = self.sc.rl_env_step(action) 

        self.render(action, reward, obs)
        return obs, reward, done, info


    def reset(self):
        # restore to initial values
        self.sc.reset()

        # initial state values for all levels in supply chain is 12
        # current state 
        self.state = self.sc.encode_tuple((12,12,12,12))
        return self.state

    def render(self, action, reward, obs):
        pass
        # print("=============================================================================")
        # print(f"t: {self.sc.t};  Action: {self.sc.action_space_tuples[action]}")
        # print(f"Reward: {reward}; Total rewards: {self.sc.total_cost}")
        # print(f"S': {obs}")