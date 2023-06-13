# multi-armed bandit problems

多腕バンディット問題の実装

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
上のコマンドを実行すると各手法の csv ファイルが csv という名前のフォルダ内に生成される.  
そして, 生成された csv ファイルを用いて各評価指標 (reward, regret, accuracy) の結果を plot する.  
plot するには以下のコードを実行
```
python plot.py csv/[フォルダ名]  # []はいらないです
```
例：`main.py` で生成されたフォルダ名が `202306081550` の場合  
```
python plot.py csv/202306081550
```
plot された図は png フォルダ内に保存される  

## アルゴリズム選択

`main.py`の`policy_list`に使用したいアルゴリズムを定義(初期状態で全てのアルゴリズムを使用)
ただし，`E_greedy`の`eps`は各自定義してください.
```
policy_list = [E_greedy(n_arms, eps=0.1), 
			   UCB1(n_arms), 
			   TS(n_arms)]
```

## 注意事項
基本的な実装しかしてないため, 必ずしも正しいという保証はありません.
