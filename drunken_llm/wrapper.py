import torch
import torch.nn as nn
from transformers import LogitsProcessorList, PreTrainedModel
from typing import Optional, Any

from .config import DrunkConfig
from .processors import DrunkenLogitsProcessor
from .memory import corrupt_kv_cache
from .rationality import RationalityManager
from .steering import SteeringManager

class DrunkenWrapper:
    """
    Hugging Face のモデルをラップし、酔い状態を付加するメインクラス。
    """
    def __init__(self, model: PreTrainedModel, config: DrunkConfig):
        self.model = model
        self.config = config
        self.rationality_manager = RationalityManager(exit_rate=config.early_exit_rate)
        self.steering_manager = SteeringManager(model)

    def generate(self, *args, **kwargs) -> Any:
        # 1. LogitsProcessor の設定（ろれつの回らなさ）
        intoxicated_gen = DrunkenLogitsProcessor(
            intoxication_level=self.config.intoxication_level,
            slur_intensity=self.config.slur_intensity
        )
        
        logits_processor = kwargs.pop("logits_processor", LogitsProcessorList())
        logits_processor.append(intoxicated_gen)
        kwargs["logits_processor"] = logits_processor

        # 2. サンプリングパラメータの調整
        if self.config.temperature_boost > 0:
            kwargs["temperature"] = kwargs.get("temperature", 1.0) + self.config.temperature_boost
            kwargs["do_sample"] = True

        # 3. 理性の減退（Early Exit）の適用
        self.rationality_manager.exit_rate = self.config.early_exit_rate
        self.rationality_manager.bypass_layers(self.model)

        # 4. 生成実行
        try:
            # kwargs に past_key_values が渡される場合の処理は、カスタムループが必要になることがあるが、
            # 通常の generate API でもフックなどは機能する。
            outputs = self.model.generate(*args, **kwargs)
        finally:
            # 5. 元の状態に復元（シラフに戻る）
            self.rationality_manager.restore_layers(self.model)

        return outputs

    def sober_up(self):
        """
        酔いを醒ます（すべてのパラメータをリセットし、フックを解除）。
        """
        self.config.apply_intoxication(0.0)
        self.rationality_manager.restore_layers(self.model)
        self.steering_manager.clear()
