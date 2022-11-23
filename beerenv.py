import sys
import random
import math
from gym import Env, spaces

from data import *
from agent import SupplyChain

class BeerGameEnv(Env):

    def __init__(self, data):
        
        # observations space coded inventory levels
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(9, start=1), 
             spaces.Discrete(9, start=1), 
             spaces.Discrete(9, start=1), 
             spaces.Discrete(9, start=1)) 
        )        
        
        # number of actions is 4^4
        self.action_space = spaces.Discrete(256) 

        # comment out vector form of actions for the mean time
        # # policy (y) values in [0,1,2,3]
        # self.action_space = spaces.Tuple(
        #     (spaces.Discrete(4, start=0), 
        #      spaces.Discrete(4, start=0), 
        #      spaces.Discrete(4, start=0), 
        #      spaces.Discrete(4, start=0)) 
        # )

        # # number of actions is 4^4
        # na = len(self.action_space)
        # self.nA = pow(na,na)

        self.sc = SupplyChain(data)

        # initial state values for all levels in supply chain is 12
        si = self.sc.encode(12)

        # current state 
        self.state = (si, si, si, si)
    


    def step(self, action):
        assert self.action_space.contains(action)
        obs, reward, done, info = self.sc.rl_env_step(action) 

        self.render(action, reward, obs)
        return obs, reward, done, info


    def reset(self):
        # restore to initial values
        self.sc.reset()

        # initial state values for all levels in supply chain is 12
        si = self.sc.encode(12)

        # current state 
        self.state = (si, si, si, si)
        return self.state

    def render(self, action, reward, obs):
        pass
        # print("=============================================================================")
        # print(f"t: {self.sc.t};  Action: {self.sc.action_space_tuples[action]}")
        # print(f"Reward: {reward}; Total rewards: {self.sc.total_cost}")
        # print(f"S': {obs}")