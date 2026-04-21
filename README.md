# drunken-llm 🥴

LLM（大規模言語モデル）の推論プロセスに構造的な介入を行い、数学的に「酔い」の状態を再現するためのライブラリです。

単なるプロンプトエンジニアリング（「酔っ払ったふりをして」）ではなく、モデルの内部演算を直接汚染することで、本物の酩酊状態に近い、予測不能で感情的な挙動を実現します。

## 🌟 主な機能

- **Logit Corruptor (ろれつの回らなさ)**: 出力単語の確率分布（ロジット）にノイズを注入し、思考の確信度を揺らします。
- **Memory Corruptor (記憶の混濁)**: KV キャッシュ（短期記憶）を意図的に破損させ、「さっき何の話してたっけ？」という忘却を再現します。
- **Rationality Reduction (理性の麻痺)**: 上位レイヤーの演算をスキップ（Early Exit）し、アライメント（自制心）を弱め、モデル本来の「本能」を露出させます。
- **Activation Steering (感情の偏り)**: 隠れ層に特定の感情ベクトルを注入し、泣き上戸、笑い上戸などの気質の変容を再現します。

## 🚀 クイックスタート

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-repo/drunken-llm.git
cd drunken-llm

# venv の作成とインストール
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 基本的な使い方

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from drunken_llm import DrunkenWrapper, DrunkConfig

# 1. 通常のモデル読み込み
model = AutoModelForCausalLM.from_pretrained("your-favorite-model")
tokenizer = AutoTokenizer.from_pretrained("your-favorite-model")

# 2. 酔いの設定
config = DrunkConfig()
config.apply_intoxication(0.6) # 酔い度 0.6 (ほろ酔い〜中酔い)

# 3. ラッパーによる「汚染」
drunk_model = DrunkenWrapper(model, config)

# 4. 生成（通常の generate API が使えます）
input_ids = tokenizer.encode("今日のご飯は何がいいかな？", return_tensors="pt")
outputs = drunk_model.generate(input_ids, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
```

## 🛠️ 技術的な背景

人間の酔いが、脳の「大脳新皮質（理性）」の活動低下により「大脳辺縁系（本能）」が優位になる現象であるのと同様に、本ライブラリでは LLM の上位推論レイヤーやアライメント層の影響を排除し、事前学習で得られた生の単語確率を前面に出すように設計されています。

## 📝 免責事項

本ライブラリは実験的・クリエイティブな目的で開発されています。
「理性のガードレール」を意図的に外す機能が含まれているため、出力の安全性や適切さが保証されない場合があります。公開環境での使用には十分ご注意ください。

## 📄 ライセンス

MIT License
