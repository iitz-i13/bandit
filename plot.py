import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import shutil

def plot_results(csv_dir_path):
    # フォルダ内のCSVファイルを読み込む
    files = os.listdir(csv_dir_path)
    policy_data = {}
    for file in files:
        if file.endswith('.csv'):
            policy_name = file.split('.')[0]
            file_path = os.path.join(csv_dir_path, file)
            df = pd.read_csv(file_path)
            policy_data[policy_name] = df

    # Rewardのプロットと保存
    plt.figure(figsize=(10, 6))
    plt.title('Reward')
    for policy_name, df in policy_data.items():
        plt.plot(df['Reward'], label=policy_name)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'reward.png'))
    plt.close()

    # Regretのプロットと保存
    plt.figure(figsize=(10, 6))
    plt.title('Regret')
    for policy_name, df in policy_data.items():
        plt.plot(df['Regret'], label=policy_name)
    plt.xlabel('Step')
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'regret.png'))
    plt.close()

    # Accuracyのプロットと保存
    plt.figure(figsize=(10, 6))
    plt.title('Accuracy')
    for policy_name, df in policy_data.items():
        plt.plot(df['Accuracy'], label=policy_name)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'accuracy.png'))
    plt.close()

time_now = datetime.now()
results_dir = 'png/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)
# 引数からCSVフォルダのパスを取得
csv_dir_path = sys.argv[1]
shutil.copyfile(csv_dir_path+"/"+"log.txt", results_dir + "/log.txt")
# 結果をプロットし、PNGファイルとして保存する
plot_results(csv_dir_path)