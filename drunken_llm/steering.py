import torch
import torch.nn as nn
from typing import List, Dict, Optional

class SteeringManager:
    """
    Activation Steering（隠れ層へのベクトル注入）を管理するクラス。
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.active_vectors: Dict[int, torch.Tensor] = {}

    def register_steering(self, layer_idx: int, vector: torch.Tensor, intensity: float = 1.0):
        """
        特定のレイヤーにステアリングベクトルを登録する。
        """
        steer_vector = vector.to(self.model.device) * intensity

        def hook_fn(module, input, output):
            # output は通常 (batch, seq, hidden_size) のテンソル
            # または Llama の場合は (output_tensor, past_key_value, ...) のタプル
            if isinstance(output, tuple):
                hidden_states = output[0]
                steered_states = hidden_states + steer_vector
                return (steered_states,) + output[1:]
            else:
                return output + steer_vector

        # 対象レイヤーの取得
        target_layer = self._get_layer(layer_idx)
        if target_layer:
            handle = target_layer.register_forward_hook(hook_fn)
            self.handles.append(handle)

    def clear(self):
        """
        すべてのフックを解除する。
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _get_layer(self, idx: int) -> Optional[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            if 0 <= idx < len(layers):
                return layers[idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
            if 0 <= idx < len(layers):
                return layers[idx]
        return None
