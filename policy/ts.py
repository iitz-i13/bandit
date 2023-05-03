import numpy as np

class TS(object):
    def __init__(self, n_arms):
        self.n_arms = n_arms # 腕の数
        self.name = 'TS'
        
        self.S = np.ones(self.n_arms)
        self.F = np.ones(self.n_arms)
        
    def initialize(self):
        self.S = np.ones(self.n_arms)
        self.F = np.ones(self.n_arms)
    
    def chosen_arm(self):
        theta = np.array([np.random.beta(self.S[i], self.F[i]) for i in range(self.n_arms)])
        max_idx = np.where(theta == max(theta))[0]
        select = np.random.choice(max_idx)
        return select
  
    def update(self, chosen_arm, reward):
        if reward == 1:
            self.S[chosen_arm] += 1
        else:
            self.F[chosen_arm] += 1