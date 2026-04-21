import argparse
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drunken_llm import DrunkenWrapper, DrunkConfig

def main():
    parser = argparse.ArgumentParser(description="Drunken LLM CLI Chat")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2", help="モデル名またはパス")
    parser.add_argument("--level", type=float, default=0.5, help="酔い度 (0.0 〜 1.0)")
    args = parser.parse_args()

    print(f"🍷 モデルを読み込んでいます: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # 酔いの設定を適用
    config = DrunkConfig()
    config.apply_intoxication(args.level)
    
    drunk_model = DrunkenWrapper(model, config)

    print(f"\n--- 🍺 Drunken LLM CLI Chat (酔い度: {args.level}) ---")
    print("'exit' または 'quit' で終了、'sober up' で酔いを醒まします。")
    
    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if user_input.lower() == "sober up":
                drunk_model.sober_up()
                print("--- 💧 酔いが醒めました！ ---")
                continue

            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            
            # 生成実行
            with torch.no_grad():
                outputs = drunk_model.generate(
                    input_ids, 
                    max_new_tokens=100, 
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 入力部分を除いて表示を試みる（簡易的）
            response = full_text[len(user_input):].strip()
            
            print(f"\n酔っ払いAI: {response}")

        except KeyboardInterrupt:
            print("\n👋 終了します。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")

    print("\n👋 バイバイ！")

if __name__ == "__main__":
    main()
