import numpy as np

class UCB1(object):
    def __init__(self, n_arms):
        self.n_arms = n_arms # 腕の数
        self.name = 'UCB1'
        
    def initialize(self):
        self.Q = np.zeros(self.n_arms) # 各腕の行動価値
        self.n = np.zeros(self.n_arms) # 各腕の行動選択回数
        self.N = np.sum(self.n) # 総試行回数
    
    def chosen_arm(self):
        for arm in range(self.n_arms):
            if self.n[arm] == 0.0: return arm
        else:
            bonus = np.sqrt((2 * np.log(self.N)) / self.n)
            ucb1 = self.Q + bonus
            max_idx = np.where(ucb1 == np.max(ucb1))[0]
            select = np.random.choice(max_idx)
            return select
  
    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        self.Q[chosen_arm] += (reward - self.Q[chosen_arm]) / self.n[chosen_arm]
        self.N += 1