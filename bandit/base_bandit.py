import numpy as np

class BaseBandit(object):
    # ベースとなるバンディットクラス
    def __init__(self, n_arms):
        self.n_arms = n_arms # 腕の数
        self.probs = np.random.rand(self.n_arms) # 各腕の真の報酬確率
        
    def pull(self, chosen_arm):
        if self.probs[chosen_arm] > np.random.rand(): # 選択した腕の真の報酬確率より大きい場合：報酬１を返す
            return 1
        else:
            return 0