# /root/hhq/main_code/utils/celf_solver.py
import torch
import heapq

class CELFSelector:
    def __init__(self, Pi, query_relevance, costs, lambda_param=1.0):
        """
        Args:
            Pi: (N, N) Reachability Matrix
            query_relevance: (N,) Tensor
            costs: (N,) Tensor
        """
        self.Pi = Pi.cpu() 
        self.rel = query_relevance.cpu()
        self.costs = costs.cpu()
        self.lambda_param = lambda_param
        self.N = len(costs)
        
    def objective_function(self, current_S_indices):
        """ 公式 (8) """
        if not current_S_indices:
            return 0.0
        
        S = torch.tensor(current_S_indices, dtype=torch.long)
        
        # Term 1: Relevance
        f_rel = torch.sum(self.rel[S]).item()
        
        # Term 2: Reachable Info Gain (Eq. 7)
        pi_sub = self.Pi[S, :] 
        sum_pi = torch.sum(pi_sub, dim=0) # sum_{v in S} Pi_{vu}
        log_term = torch.log(1 + sum_pi)
        f_reach = torch.sum(self.rel * log_term).item()
        
        return f_rel + self.lambda_param * f_reach

    def select(self, budget):
        """ CELF 算法主逻辑 """
        S = []
        current_cost = 0
        current_obj = 0.0
        
        pq = [] 
        # 1. Init: Calculate marginal gain for all singletons
        for v in range(self.N):
            cost_v = self.costs[v].item()
            if cost_v > budget: continue
            
            gain = self.objective_function([v])
            mg = gain / cost_v
            heapq.heappush(pq, (-mg, cost_v, v)) # Min-heap
            
        # 2. Loop
        while pq:
            neg_gain, cost, u = heapq.heappop(pq)
            
            if current_cost + cost > budget:
                continue
                
            # Recompute gain
            new_obj = self.objective_function(S + [u])
            real_gain = new_obj - current_obj
            real_mg = real_gain / cost
            
            if not pq:
                S.append(u)
                current_cost += cost
                current_obj = new_obj
                break
                
            # Check against next best
            best_peer_neg_mg, _, _ = pq[0]
            if real_mg >= (-best_peer_neg_mg):
                S.append(u)
                current_cost += cost
                current_obj = new_obj
            else:
                # Push back
                heapq.heappush(pq, (-real_mg, cost, u))
                
        return sorted(S)