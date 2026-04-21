import torch
import random
from typing import Tuple, Optional

def corrupt_kv_cache(
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    memory_loss_rate: float = 0.1
) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
    """
    KVキャッシュを意図的に破損またはノイズを混入させることで、短期記憶の欠落を再現する。
    
    Args:
        past_key_values: モデルの過去のコンテキスト（キーと値のペア）
        memory_loss_rate: 記憶を失う確率（0.0〜1.0）
    """
    if past_key_values is None or memory_loss_rate <= 0:
        return past_key_values
    
    corrupted_kv = []
    for layer_kv in past_key_values:
        # layer_kv は通常 (key, value) のペア
        keys, values = layer_kv
        
        # 記憶喪失率を試行
        if random.random() < memory_loss_rate:
            # テンソルの一部をゼロアウト、またはノイズを加える
            # ここでは微小なノイズを加えて「記憶の混濁」を表現する
            noise_scale = 0.02 * memory_loss_rate
            keys = keys + torch.randn_like(keys) * noise_scale
            values = values + torch.randn_like(values) * noise_scale
            
        corrupted_kv.append((keys, values))
        
    return tuple(corrupted_kv)
