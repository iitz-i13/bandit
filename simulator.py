import os
import numpy as np
import pandas as pd
from datetime import datetime
from bandit.base_bandit import BaseBandit
from policy.e_greedy import E_greedy
from policy.ucb1 import UCB1
from policy.ts import TS
import matplotlib.pyplot as plt

class BanditSimulator(object):
    def __init__(self, policy_list, n_sims, n_steps, n_arms):
        self.policy_list = policy_list
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.n_arms = n_arms
        self.output_folder = os.path.join(os.path.dirname(__file__), 'csv', datetime.now().strftime("%Y%m%d%H%M"))
        
        self.select = np.zeros((self.n_sims, self.n_steps))
        self.regret = np.zeros((self.n_sims, self.n_steps))
        self.reward = np.zeros((self.n_sims, self.n_steps))
        self.accuracy = np.zeros((self.n_sims, self.n_steps))
        self.total_reward = np.zeros((self.n_sims, self.n_steps))
        
    def run(self):
        os.makedirs(self.output_folder, exist_ok=True)
        
        for policy in self.policy_list:
            print(policy.name)
            for sim in range(self.n_sims):
                self.env = BaseBandit(self.n_arms)
                self.probs = self.env.probs
                opt_arm = np.where(self.probs == max(self.probs))[0]
                sort_probs = sorted(self.probs)
                first = sort_probs[-1]
                second = sort_probs[-2]
                aleph = (first + second) / 2
                
                policy.initialize()
                
                for step in range(self.n_steps):
                    chosen_arm = policy.chosen_arm()
                    reward = self.env.pull(chosen_arm)
                    policy.update(chosen_arm, reward)
                    
                    self.reward[sim, step] = reward
                    self.select[sim, step] = chosen_arm
                    if step == 0:
                        self.regret[sim, step] = self.probs[opt_arm[0]] - self.probs[chosen_arm]
                    else:
                        self.regret[sim, step] = self.regret[sim, step-1] + (self.probs[opt_arm[0]] - self.probs[chosen_arm])
                    
                    # Calculate accuracy
                    if chosen_arm == opt_arm[0]:
                        self.accuracy[sim, step] = 1.0
                    else:
                        self.accuracy[sim, step] = 0.0
                    
                    # Calculate total reward
                    self.total_reward[sim, step] = np.sum(self.reward[sim, :step+1])

            self.save_data(policy)

    def save_data(self, policy):
        data = {
            'Reward': np.mean(self.reward, axis=0),
            'Regret': np.mean(self.regret, axis=0),
            'Accuracy': np.mean(self.accuracy, axis=0),
            'Total Reward': np.mean(self.total_reward, axis=0)
        }
        df = pd.DataFrame(data)
        filename = f"{policy.name}.csv"
        filepath = os.path.join(self.output_folder, filename)
        df.to_csv(filepath, index=False)