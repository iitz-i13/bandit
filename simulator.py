import numpy as np
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
        
        self.select = np.zeros((self.n_sims, self.n_steps))
        self.regret = np.zeros((self.n_sims, self.n_steps))
        self.reward = np.zeros((self.n_sims, self.n_steps))
        
    def run(self):
        for policy in self.policy_list:
            print(policy.name)
            for sim in range(self.n_sims):
                self.env = BaseBandit(self.n_arms)
                self.probs = self.env.probs
                # print('self.probs: ', self.probs)
                opt_arm = np.where(self.probs == max(self.probs))[0]
                # print('opt_arm: ', opt_arm[0])
                sort_probs = sorted(self.probs)
                first = sort_probs[-1]
                second = sort_probs[-2]
                aleph = (first + second) / 2
                
                policy.initialize()
                
                for step in range(self.n_steps):
                    chosen_arm = policy.chosen_arm()
                    # print('chosen_arm: ',chosen_arm)
                    reward = self.env.pull(chosen_arm)
                    policy.update(chosen_arm, reward)
                    
                    self.reward[sim, step] = reward
                    self.select[sim, step] = chosen_arm
                    if step == 0:
                        self.regret[sim,step] = self.probs[opt_arm[0]] - self.probs[chosen_arm]
                    else:
                        self.regret[sim,step] = self.regret[sim, step-1] + (self.probs[opt_arm[0]] - self.probs[chosen_arm])   
                        
            self.plot_regret(policy)
        self.print_regret()
    
    def plot_regret(self, policy):
        plt.plot(np.arange(self.n_steps), np.mean(self.regret, axis=0), label=policy.name)
    
    def print_regret(self):
        plt.xlabel('step')
        plt.ylabel('regret')
        plt.legend()
        plt.show()