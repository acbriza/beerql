import sys
import random
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
        
        # policy (y) values in [0,1,2,3]
        self.action_space = spaces.Tuple(
            (spaces.Discrete(4, start=0), 
             spaces.Discrete(4, start=0), 
             spaces.Discrete(4, start=0), 
             spaces.Discrete(4, start=0)) 
        )           

        self.sc = SupplyChain(data=data[MAIN], policy=RLCha08)

        # initial state values for all levels in supply chain is 12
        si = self.sc.encode(12)

        # current state 
        self.state = (si, si, si, si)
    


    def step(self, action):
        assert self.action_space.contains(action)
        obs, reward, done, info = sc.step(action) 

        self.render(action, reward, obs)
        return obs, reward, done, info


    def reset(self):
        # restore to initial values
        
        # initial state values for all levels in supply chain is 12
        si = self.sc.encode(12)

        # current state 
        self.state = (si, si, si, si)

    def render(self, action, reward, obs):
        print("=============================================================================")
        print(f"t: {self.sc.t};  Action: {action}")
        print(f"Reward: {reward}; Total rewards: {self.sc.total_cost}")
        print(f"S': {obs}")

        # print(f"Action : {action}")
        # print(f"Total Reward : {self.collected_reward}")
