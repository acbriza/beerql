import sys

from data import DEMAND, LEAD, TIME_HORIZON
import numpy as np

class SCAgent():

    get_cost = lambda x: x if x > 0 else -2*x

    def __init__(self, index, name, init_inventory=0, forwarded_orders=[]):
        """ init_inv - int:  initial inventory
            forwarded_orders - tuple of (time_step, order size)
        """
        self.t = 0
        self.name = name
        self.index = index
        self.txn = {
            "received": {t:0 for t in range(TIME_HORIZON+1)},
            "inventory": {t:0 for t in range(TIME_HORIZON+1)},
            "policy": {t:0 for t in range(TIME_HORIZON+1)},
            "lead": {t:0 for t in range(TIME_HORIZON+1)},
            "orders" : {t:[] for t in range(0, TIME_HORIZON+1)},
            "forwarded": {t:0 for t in range(TIME_HORIZON+1)},
        }

        # set initial values        
        self.txn["inventory"][0] = init_inventory
        self.txn["policy"][0] = np.NaN
        self.txn["lead"][0] = np.NaN
        for t, order in forwarded_orders:
            self.txn["forwarded"][t] += order

    def step(self):
        self.t += 1

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
                forwarded = prev_inv 
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

    def post_order(self, demand, y, lead):
        if self.t + lead >= TIME_HORIZON:
            self.orders[self.t + lead].append(demand+y)


class SupplyChain():
            
    def __init__(self, data, policy):
        self.t = 0
        self.data = data
        self.policy = policy
        self.agents = []
        self.agents.append(SCAgent(1, "Retailer", 12, []))
        self.agents.append(SCAgent(2, "Distributor", 12, [(0,4), (1,4)]))
        self.agents.append(SCAgent(3, "Manufacturer", 12, []))
        self.agents.append(SCAgent(4, "Supplier", 12, []))
        self.agents.append(SCAgent(5, "Source", 0, []))
        self.NUM_AGENTS = len(self.agents)
        # add demand values to retailer agent
        self.agents[0].add_order_list(self.data[DEMAND]) 

    def print_info(self):
        print(f"t:{self.t} ", end="")
        for agent in self.agents:
            print(agent.info())

    def get_report(self, steps):
        report = {}
        for agent in self.agents:
            report[agent.name] = {
            "received": list(agent.txn["received"].values())[:steps+1],
            "inventory": list(agent.txn["received"].values())[:steps+1],
            "policy": list(agent.txn["policy"].values())[:steps+1],
            "lead": list(agent.txn["lead"].values())[:steps+1],
            "orders": list(agent.txn["orders"].values())[:steps+1],
            "forwarded": list(agent.txn["forwarded"].values())[:steps+1],
        }
        return report

    def step(self, verbosity=0):
        self.t += 1
        if self.t > TIME_HORIZON:
            print('Finished epoch')
            return

        for agent in self.agents:
            agent.step()

        # receive deliveries
        for i in [3,2,1,0]:
            delivery = self.agents[i+1].get_forwarded_deliveries()
            self.agents[i].receive_delivery(delivery)  
        
        # process orders 
        lead = self.data[LEAD][self.t-1]
        y = self.policy[self.t-1]
        for i in [0,1,2,3]: 
            # process orders from downstream
            demand = self.agents[i].process_orders()
            if demand > 0:
                # post order to upstream
                self.agents[i+1].post_order(demand, y, lead)

        self.agents[4].process_orders_source()
    
    def run(self, steps=TIME_HORIZON):
        for _ in range(steps):
            self.step(verbosity=1)
        return self.get_report(steps)
