# multi-armed bandit
多腕バンディット問題の実装

# Requirements
- python 3.11.3
- numpy 1.23.5
- pandas 2.0.0
- matplotlib 3.7.1

## 実装済みのアルゴリズム

- (ε)Greedy ( ε は調整可能)
- Upper Confidence Bound 1 (UCB1)
- Thompson sampling (TS)

## 使い方（Usage）
`main.py`で, 実験設定を行なってください．
```
def main():
    n_arms = 3 # 腕の数
    n_sims = 100 # シミュレーション数
    n_steps = 1000 # step数
```

実行するには以下のコード
```bash
python main.py
```  
上のコマンドを実行すると各手法の csv ファイルとどのような設定で実験を行ったのかという log.txt が csv という名前のフォルダ内に生成される.  
log.txt 内は以下のような内容が記録される.  
```
policy_name: ['Greedy', 'UCB1', 'TS']
dataset inf: 
csv: 202306081550
n_arms: 3
sim: 100, step: 1000
```
生成された csv ファイルを用いて各評価指標 (reward, regret, accuracy) の結果を plot するには以下のコードを実行.
```
python plot.py csv/[フォルダ名]  # []はいらないです
```
例：`main.py` で生成されたフォルダ名が `202306081550` の場合  
```
python plot.py csv/202306081550
```
plot された図は png フォルダ内に保存される  

## アルゴリズム選択

`main.py` の `policy_list` に使用したいアルゴリズムを定義(初期状態で全てのアルゴリズムを使用)
ただし, `E_greedy` の `eps` は各自定義してください.
```
policy_list = [E_greedy(n_arms, eps=0.1), 
			   UCB1(n_arms), 
			   TS(n_arms)]
```