from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drunken_llm import DrunkenWrapper, DrunkConfig

def main():
    # 軽量なモデルでデモ（実際の運用では Llama-3 などを使用推奨）
    model_name = "sshleifer/tiny-gpt2" 
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "Today I went to the"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # 1. シラフ状態での生成
    print("\n--- 状態: シラフ (Sober) ---")
    outputs = model.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # 2. 泥酔状態での生成
    print("\n--- 状態: 泥酔 (Drunk 0.8) ---")
    config = DrunkConfig()
    config.apply_intoxication(0.8) # 酔い度 0.8
    
    drunk_model = DrunkenWrapper(model, config)
    
    # DrunkenWrapper.generate を使用（内部でロジット汚染やレイヤースキップが行われる）
    outputs = drunk_model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # 3. 酔いを醒ます
    print("\n--- 状態: 酔い醒まし (Sobered up) ---")
    drunk_model.sober_up()
    outputs = drunk_model.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
