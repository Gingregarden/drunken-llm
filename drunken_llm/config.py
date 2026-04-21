from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class DrunkConfig:
    """
    酔い状態の設定を管理するクラス。
    """
    intoxication_level: float = 0.0  # 0.0 (シラフ) 〜 1.0 (泥酔)
    
    # 個別のモジュールの強度（intoxication_level に連動させることも可能）
    memory_loss_rate: float = 0.0    # KVキャッシュを欠落させる確率
    slur_intensity: float = 0.0      # ロジットへのノイズ注入強度
    temperature_boost: float = 0.0   # ベースの Temperature への上乗せ分
    early_exit_rate: float = 0.0     # 上位レイヤーをスキップする割合 (0.0〜1.0)
    
    # 感情ステアリング
    emotion_vector_path: Optional[str] = None
    emotion_intensity: float = 0.0
    
    # プリセット用のスタイル
    style: Optional[str] = None  # "talkative", "cheerful", "crying", "angry" etc.

    def apply_intoxication(self, level: float):
        """
        酔い度を一括で設定し、各パラメータを自動調整する。
        """
        self.intoxication_level = level
        self.memory_loss_rate = level * 0.3
        self.slur_intensity = level * 1.5
        self.temperature_boost = level * 1.0
        self.early_exit_rate = level * 0.5
