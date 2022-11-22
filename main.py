import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from data import *
from agent import SCAgent, SupplyChain
sc = SupplyChain(data=data[MAIN], policy=RL)
report = sc.run(5)

df = pd.DataFrame.from_dict(report, orient="index").stack().to_frame()
# to break out the lists into columns
df = pd.DataFrame(df[0].values.tolist(), index=df.index)
print(df.T)