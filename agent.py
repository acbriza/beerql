import sys

from data import DEMAND, LEAD

class SCAgent():

    def append_order(self, demand, lead):
        while lead > len(self.orders):
            self.orders.append([])
        self.orders[lead-1].append(demand)

    def __init__(self, index, name, init_inv=0, init_orders=[]):    
        """ init_inv - int:  initial inventory
            orders - tuple of (order size, lead): orders in transit
        """
        self.name = name
        self.index = index
        self.inv = init_inv
        self.orders = []
        for demand,lead in init_orders :
            self.append_order(demand, lead)
        self.inventory = [init_inv, ]
        self.delivery = [-sys.maxsize-1]
        self.demand = [-sys.maxsize-1]
        self.policy = [-sys.maxsize-1]
        self.lead = [-sys.maxsize-1]

            
    def _get_current_order(self):
        if self.orders:
            total_orders = sum(self.orders[0]) 
        else:
            total_orders = 0
        return total_orders
        
    def info(self):
        return f"Inv:{self.inv}; Order:{self._get_current_order()}"

    def step(self, demand, lead, y):
        """ y represents the x+y policy
            returns the 

        """
        # receive orders
        if self.orders:
            delivered_order = sum(self.orders.pop(0)) 
        else:
            delivered_order = 0        
        # satisfy downstream
        self.inv = self.inv + delivered_order - demand
        # place an order
        self.append_order(demand+y, lead)            
        # this delivered_order becomes a demand for the upstream
        upstream_demand = delivered_order

        # update statistics
        self.inventory.append(self.inv)
        self.delivery.append(delivered_order)
        self.demand.append(demand)
        self.policy.append(y)
        self.lead.append(lead)
        return upstream_demand


class SupplyChain():
            
    def __init__(self, data, time_horizon, policy):
        self.t = 0        
        self.data = data        
        self.policy = policy
        self.time_horizon = time_horizon
        self.agents = []
#         self.agents.append(SCAgent(0, "Customer", 0, []))
        self.agents.append(SCAgent(1, "Retailer", 12, [(4,1), (4,2)]))
        self.agents.append(SCAgent(2, "Distributor", 16, []))
        self.agents.append(SCAgent(3, "Manufacturer", 12, []))
        self.agents.append(SCAgent(4, "Supplier", 12, []))

    def print_info(self):
        print(f"t:{self.t} ", end="")
        for agent in self.agents:
            print(agent.info())
    
    def step(self):
        # self.print_info()
        self.t += 1
        if self.t >= self.time_horizon:
            print('Finished epoch')
        demand = self.data[DEMAND][self.t-1]
        lead = self.data[LEAD][self.t-1]
        y = self.policy[self.t-1]
        #print(demand, lead)
        for agent in self.agents:            
            demand = agent.step(demand, lead, y)
    
    def run(self, steps=None):
        "If steps is None, run everything"
        time_steps = range(steps) if steps else range(self.time_horizon) 
        for t in time_steps:
            print(t)
            self.step()

        report ={}
        for agent in self.agents:
            report[agent.name] = {
                "inventory": agent.inventory,
                "delivery": agent.delivery,
                "demand": agent.demand,
                "policy": agent.policy,
                "lead": agent.lead,
            }    
        return report
        

