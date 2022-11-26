import sys

from data import DEMAND, LEAD, TIME_HORIZON
import numpy as np
import pandas as pd
from itertools import product

class SCAgent():

    def reset(self):
        self.t = 0
        self.txn = {
            "received": {t:0 for t in range(TIME_HORIZON+1)},
            "inventory": {t:0 for t in range(TIME_HORIZON+1)},
            "policy": {t:0 for t in range(TIME_HORIZON+1)},
            "lead": {t:0 for t in range(TIME_HORIZON+1)},
            "orders" : {t:[] for t in range(0, TIME_HORIZON+1)},
            "forwarded": {t:0 for t in range(TIME_HORIZON+1)},
        }

        # set initial values        
        self.txn["inventory"][0] = self.init_inventory
        self.txn["policy"][0] = np.NaN
        self.txn["lead"][0] = np.NaN
        for t, order in self.init_forwarded_orders:
            self.txn["forwarded"][t] += order

    def __init__(self, index, name, init_inventory=0, init_forwarded_orders=[]):
        """ init_inv - int:  initial inventory
            init_forwarded_orders - tuple of (time_step, order size)
        """
        self.name = name
        self.index = index
        self.init_inventory = init_inventory
        self.init_forwarded_orders = init_forwarded_orders
        self.txn = {}
        self.reset()

    def step(self, policy, lead):
        self.t += 1
        self.txn["policy"][self.t] = policy
        self.txn["lead"][self.t] = lead

    def add_order_list(self, order_list):
        assert len(order_list) == TIME_HORIZON
        for i, order in enumerate(order_list):
            self.txn["orders"][i+1].append(order)

    def append_order(self, demand, lead):
        if self.t+lead <= TIME_HORIZON:            
            self.orders[self.t+lead].append(demand)

    def get_forwarded_deliveries(self):
        "Return forwarded items from previous time step"
        assert self.t-1 >= 0, f"Invalid index in function get_forwarded_deliveries: {self.t-1}"
        return self.txn["forwarded"][self.t-1]

    def get_current_total_orders(self):
        orders = self.txn["orders"][self.t]
        return sum(orders) if orders else 0

    def receive_delivery(self, delivery):
        self.txn["received"][self.t] += delivery
        
    def process_orders(self):
        """ Assumes delivery has already been received
            Updates current inventory as well as forwarded stocks
            Returns demand received for processing upstream
        """
        received = self.txn["received"][self.t]
        prev_inv = self.txn["inventory"][self.t-1]
        demand = self.get_current_total_orders()

        # Fulfill backorders
        if prev_inv < 0:
            if received >= abs(prev_inv):
                # completely fulfill backorder
                forwarded = abs(prev_inv)
            else:
                # partially fulfill backorder with received goods
                forwarded = received
        else:
            # we have not forwarded anything yet 
            forwarded = 0

        onhand = received + prev_inv        

        if onhand > 0 and demand > 0:
            # we can also serve the new demand after having fulfilled back order, if any
            if onhand - demand >= 0:
                # we have enough physical stocks to make the full delivery
                forwarded += demand
            else:
                # not enough stock; just deliver what we have on hand
                forwarded += onhand

        # set forwarded orders
        self.txn["forwarded"][self.t] += forwarded
 
        # update inventory for this time step
        self.txn["inventory"][self.t] = onhand - demand
    
        return demand

    def process_orders_source(self):
        """process orders for first in the supply chain (prior to supplier)"""
        demand = self.get_current_total_orders()
        self.txn["forwarded"][self.t] = demand

    def post_order(self, demand, policy, lead):
        y = policy
        idx = self.t + lead
        if idx <= TIME_HORIZON:
            self.txn["orders"][idx].append(demand+y)


class SupplyChain():
            
    def reset(self):
        self.t = 0        
        self.total_cost = 0
        self.cost = {t:0 for t in range(TIME_HORIZON+1)}
        self.cost[0] = np.NaN
        for agent in self.agents:
            agent.reset()        
        # add demand values to retailer agent
        self.agents[0].add_order_list(self.data[DEMAND]) 

    def __init__(self, data, policy=None):
        "policy is used for simulations;  for RL training it is set to None"
        self.data = data
        self.policy = policy
        self.agents = []
        self.agents.append(SCAgent(1, "Retailer", 12, []))
        self.agents.append(SCAgent(2, "Distributor", 12, [(0,4), (1,4)]))
        self.agents.append(SCAgent(3, "Manufacturer", 12, []))
        self.agents.append(SCAgent(4, "Supplier", 12, []))
        self.agents.append(SCAgent(5, "Source", 0, []))
        self.n_ACTING_AGENTS = len(self.agents)-1 # number of agents minus Source
        self.reset()

    action_space_tuples = tuple(product((0,1,2,3), repeat=4))

    # define a dictionary for mapping a tuple to an index in the action vector
    def iA(self, action_tuple):
        return self.action_space_tuples.index(action_tuple)

    def encode_value(self, inv):
        if inv < -6:
            return 1
        elif inv < -3:
            return 2
        elif inv < 0:
            return 3
        elif inv < 3:
            return 4
        elif inv < 6:
            return 5
        elif inv < 10:
            return 6
        elif inv < 15:
            return 7
        elif inv < 20:
            return 8
        else:
            return 9

    def encode_tuple(self, inv_tuple):
        return tuple(self.encode_value(v) for v in inv_tuple)

    def print_info(self):
        print(f"t:{self.t} ", end="")
        for agent in self.agents:
            print(agent.info())

    def get_report(self, steps):
        report = {}
        for agent in self.agents:
            report[agent.name] = {
            "received": list(agent.txn["received"].values())[:steps+1],
            "inventory": list(agent.txn["inventory"].values())[:steps+1],
            "policy": list(agent.txn["policy"].values())[:steps+1],
            "lead": list(agent.txn["lead"].values())[:steps+1],
            "orders": list(agent.txn["orders"].values())[:steps+1],
            "forwarded": list(agent.txn["forwarded"].values())[:steps+1],
        }
        report["Cost"] = {"cost": list(self.cost.values())[:steps+1],}
        return report

    def get_report_df(self, report):
        df = pd.DataFrame.from_dict(report, orient="index").stack().to_frame()
        # to break out the lists into columns
        df = pd.DataFrame(df[0].values.tolist(), index=df.index)
        return df

    def get_rl_env_step_info(self, msg=""):
        info = {}
        info["Timestep"] = self.t
        info["Cost"] = self.cost[self.t]
        info["Msg"] = msg
        info["Agents"] = {}
        for agent in self.agents:
            info["Agents"][agent.name] = {
            "received": agent.txn["received"][self.t],
            "inventory": agent.txn["inventory"][self.t],
            "policy": agent.txn["policy"][self.t],
            "lead": agent.txn["lead"][self.t],
            "orders": agent.txn["orders"][self.t],
            "forwarded": agent.txn["forwarded"][self.t],
        }        
        return info

    def update_cost(self):
        "updates cost dictionary"
        # if self.t == 0:
        #     # no need to compute cost for time step 0
        #     return
        assert self.t <= TIME_HORIZON, f"Can't compute cost for {self.t}"
        get_cost = lambda x: x if x > 0 else -2*x
        cost  = 0
        for i in [0,1,2,3]: #"Retailer",  "Distributor", "Manufacturer", "Supplier"
            cost += get_cost(self.agents[i].txn["inventory"][self.t])
        self.cost[self.t] = cost
        return cost

    def compute_cost(self):
        "computes accumulated cost up to current time step"
        return sum(list(self.cost.values()))

    def simulation_step(self, verbosity=0):        
        self.t += 1

        # get lead and policy data
        lead = self.data[LEAD][self.t-1]        
        for agent in self.agents:
            policy = self.policy[agent.index][self.t-1]
            agent.step(policy, lead)

        # receive deliveries
        for i in [3,2,1,0]:
            delivery = self.agents[i+1].get_forwarded_deliveries()
            self.agents[i].receive_delivery(delivery)  
        
        # process orders 
        for i in [0,1,2,3]: 
            # process orders from downstream
            demand = self.agents[i].process_orders()
            if demand > 0:
                # post order to upstream
                policy = self.agents[i].txn["policy"][self.t]
                self.agents[i+1].post_order(demand, policy, lead)

        self.agents[4].process_orders_source()

        # compute cost 
        cost = self.update_cost()
        self.total_cost += cost
        
        if self.t+1 > TIME_HORIZON:
            return

    def simultation_run(self, steps=TIME_HORIZON):
        for _ in range(steps):
            self.simulation_step()
        report = self.get_report(steps)
        df = self.get_report_df(report)
        return df

    def rl_env_step(self, action, discretized_states=True):
        """ action: integer 
                An index representing an action
            disctredized_states: boolean
                Whether the tuples representing states have to be discretized
        """

        # convert action index parameter to a 4-tuple
        action = self.action_space_tuples[action]
        
        done = False
        self.t += 1

        # get lead and policy data
        lead = self.data[LEAD][self.t-1] # data's index start at zero while t starts at 1
        for i,agent in enumerate(self.agents): 
            policy = action[i] if i<self.n_ACTING_AGENTS else np.NaN
            agent.step(policy, lead)

        # receive deliveries
        for i in [3,2,1,0]:
            delivery = self.agents[i+1].get_forwarded_deliveries()
            self.agents[i].receive_delivery(delivery)  
        
        # process orders 
        for i in [0,1,2,3]: 
            # process orders from downstream
            demand = self.agents[i].process_orders()
            if demand > 0:
                # post order to upstream
                policy = action[i]
                self.agents[i+1].post_order(demand, policy, lead)

        self.agents[4].process_orders_source()

        if discretized_states:
            # for Q-Learning, new state has to be encoded
            state = (
                self.encode_value(
                    self.agents[i].txn["inventory"][self.t]
                ) for i in [0,1,2,3]
            )
            state = tuple(state)
        else:
            state = (self.agents[i].txn["inventory"][self.t] for i in [0,1,2,3])


        # compute cost 
        cost = self.update_cost()
        self.total_cost += cost

        reward = -1*cost

        if self.t+1 > TIME_HORIZON:
            done = True
        
        info = self.get_rl_env_step_info()
            
        return state, reward, done, info
            
