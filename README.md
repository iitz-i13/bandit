# multi-armed bandit problems

多腕バンディット問題の実装 (現在、regret の出力だけ可能)

## 実装済みのアルゴリズム

- (ε)Greedy ( ε は調整可能)
- Upper Confidence Bound 1 (UCB1)
- Thompson sampling (TS)

## 使用方法
`main.py`で，実験設定を行なってください．
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

## アルゴリズム選択

`main.py`の`policy_list`に使用したいアルゴリズムを定義(初期状態で全てのアルゴリズムを使用)
ただし，`E_greedy`の`eps`は各自定義してください．
```
policy_list = [E_greedy(n_arms, eps=0.1), 
			   UCB1(n_arms), 
			   TS(n_arms)]
```

## 注意事項

`regret`の出力のみの実装であるため，お気をつけて
