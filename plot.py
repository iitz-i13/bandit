import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys

def create_results_folder():
    # 現在の日時を取得してフォルダ名とする
    now = datetime.now()
    folder_name = now.strftime('%Y%m%d%H%M')

    # 結果保存用のフォルダを作成
    os.makedirs(os.path.join('png', folder_name))

    return folder_name

def plot_results(csv_folder_path):
    output_folder_path = create_results_folder()

    # フォルダ内のCSVファイルを読み込む
    files = os.listdir(csv_folder_path)
    policy_data = {}
    for file in files:
        if file.endswith('.csv'):
            policy_name = file.split('.')[0]
            file_path = os.path.join(csv_folder_path, file)
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
    plt.savefig(os.path.join('png', output_folder_path, 'reward.png'))
    plt.close()

    # Regretのプロットと保存
    plt.figure(figsize=(10, 6))
    plt.title('Regret')
    for policy_name, df in policy_data.items():
        plt.plot(df['Regret'], label=policy_name)
    plt.xlabel('Step')
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig(os.path.join('png', output_folder_path, 'regret.png'))
    plt.close()

    # Accuracyのプロットと保存
    plt.figure(figsize=(10, 6))
    plt.title('Accuracy')
    for policy_name, df in policy_data.items():
        plt.plot(df['Accuracy'], label=policy_name)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('png', output_folder_path, 'accuracy.png'))
    plt.close()

# 引数からCSVフォルダのパスを取得
csv_folder_path = sys.argv[1]

# 結果をプロットし、PNGファイルとして保存する
plot_results(csv_folder_path)
