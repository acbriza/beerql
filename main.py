from data import *
from agent import SCAgent, SupplyChain

import numpy as np
import pandas as pd

from beerenv import BeerGameEnv
from ql import q_learning
import plotting 

def simulation():
    sc = SupplyChain(data=data[MAIN], policy=RL)
    report = sc.run(5)

    df = pd.DataFrame.from_dict(report, orient="index").stack().to_frame()
    # to break out the lists into columns
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    print(df.T)

def rl_train():
    env = BeerGameEnv()
    Q, stats = q_learning(env, num_episodes=1000, alpha=0.17)
    return Q, stats


Q, stats = rl_train()

plotting.plot_rewards(stats)