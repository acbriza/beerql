import pandas as pd
import random

DEMAND, LEAD  = 0, 1
MAIN, PROB1, PROB2, PROB3 = 0, 1, 2, 3
TIME_HORIZON = 35
data = { 
    MAIN: { 
        DEMAND: [15,10,8,14,9,3,13,2,13,11,3,4,6,11,15,12,15,4,12,3,13,10,15,15,3,11,1,13,10,10,0,0,8,0,14],
        LEAD: [2,0,2,4,4,4,0,2,4,1,1,0,0,1,1,0,1,1,2,1,1,1,4,2,2,1,4,3,4,1,4,0,3,3,4]
    },
    PROB1: {  
        DEMAND: [5,14,14,13,2,9,5,9,14,14,12,7,5,1,13,3,12,4,0,15,11,10,6,0,6,6,5,11,8,4,4,12,13,8,12],
        LEAD: [2,0,2,4,4,4,0,2,4,1,1,0,0,1,1,0,1,1,2,1,1,1,4,2,2,1,4,3,4,1,4,0,3,3,4]
    },
    PROB2: {
        DEMAND: [15,10,8,14,9,3,13,2,13,11,3,4,6,11,15,12,15,4,12,3,13,10,15,15,3,11,1,13,10,10,0,0,8,0,14],
        LEAD: [4,2,2,0,2,2,1,1,3,0,0,3,3,3,4,1,1,1,3,0,4,2,3,4,1,3,3,3,0,3,4,3,3,0,3]
    },
    PROB3: {
        DEMAND: [13,13,12,10,14,13,13,10,2,12,11,9,11,3,7,6,12,12,3,10,3,9,4,15,12,7,15,5,1,15,11,9,14,0,4],
        LEAD: [4,2,2,0,2,2,1,1,3,0,0,3,3,3,4,1,1,1,3,0,4,2,3,4,1,3,3,3,0,3,4,3,3,0,3]
    }
}

PASS_ORDER_SINGLE = [0] * 35

PASS_ORDER = {
    1: [0] * 35,
    2: [0] * 35,
    3: [0] * 35,
    4: [0] * 35,
    5: [0] * 35,
}

RLDummy = {
    1: [3,3,1,1,2] + [0]*30,
    2: [1,0,0,0,1] + [0]*30,
    3: [3,2,0,2,1] + [0]*30,
    4: [3,0,2,0,0] + [0]*30,
    5: [0]*35,
}

df = pd.read_excel("data\Cha08Solution.xlsx")
RLCha08 = {
    1: list(df.Retailer.values),
    2: list(df.Distributor.values),
    3: list(df.Manufacturer.values),
    4: list(df.Supplier.values),
    5: [0]*35,
}

def get_uniform(choices, size, seed):
    "returns a list of items from choices drawn with uniform probability"
    random.seed(seed)
    return [random.choice(choices) for i in range(size)]

RAND_UNIFORM = {
    1: get_uniform((0,1,2,3), 35, 1),
    2: get_uniform((0,1,2,3), 35, 2),
    3: get_uniform((0,1,2,3), 35, 3),
    4: get_uniform((0,1,2,3), 35, 4),
    5: get_uniform((0,1,2,3), 35, 5)
}