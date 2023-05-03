from simulator import BanditSimulator
from policy.e_greedy import E_greedy
from policy.ucb1 import UCB1
from policy.ts import TS

def main():
    n_arms = 3 # 腕の数
    n_sims = 100 # シミュレーション数
    n_steps = 1000 # step数
    
    policy_list = [E_greedy(n_arms, eps=0.1), 
                   UCB1(n_arms), 
                   TS(n_arms)]
    
    bandit_sim = BanditSimulator(policy_list, n_sims, n_steps, n_arms)
    bandit_sim.run()
    
if __name__ == '__main__':
    print('--- started run ---')
    main()
    print('--- finished run ---')