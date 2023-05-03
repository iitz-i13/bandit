import numpy as np

class E_greedy(object):
    def __init__(self, n_arms, eps):
        self.eps = eps
        self.n_arms = n_arms # 腕の数
        if self.eps == 1:
            self.name = 'Greedy'
        else:
            self.name = 'ε-greedy ε=' + str(self.eps)
        
    def initialize(self):
        self.Q = np.zeros(self.n_arms) # 各腕の行動価値
        self.n = np.zeros(self.n_arms) # 各腕の行動選択回数
    
    def greedy(self):
        max_idx = np.where(self.Q == np.max(self.Q))[0] # 行動価値Qが最大の選択肢(index)を返す
        select = np.random.choice(max_idx) # 最大の選択肢が複数ある場合はランダムに選択
        return select
    
    def chosen_arm(self):
        if np.random.rand() < self.eps: # epsの確率でランダムに行動する
            select = np.random.randint(0,self.n_arms)
        else:
            select = self.greedy()
        return select
  
    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        self.Q[chosen_arm] += (reward - self.Q[chosen_arm]) / self.n[chosen_arm]