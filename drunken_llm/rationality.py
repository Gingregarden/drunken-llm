import torch
import torch.nn as nn
from typing import Optional

class RationalityManager:
    """
    「理性（アライメントや上位レイヤーでの推論）」を減退させるためのマネージャー。
    """
    def __init__(self, exit_rate: float = 0.0):
        self.exit_rate = exit_rate
        self._original_layers = None

    def bypass_layers(self, model: nn.Module):
        """
        モデルの上位レイヤーを一時的にスキップし、理性を麻痺させる。
        """
        if self.exit_rate <= 0:
            return

        # レイヤーのリストを取得（モデル構造に依存）
        layers_attr = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_attr = model.model
            layers_name = "layers"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers_attr = model.transformer
            layers_name = "h"
        
        if layers_attr:
            full_layers = getattr(layers_attr, layers_name)
            if self._original_layers is None:
                self._original_layers = full_layers

            num_layers = len(full_layers)
            exit_index = int(num_layers * (1.0 - self.exit_rate))
            exit_index = max(1, min(exit_index, num_layers))
            
            # レイヤーを切り詰める
            setattr(layers_attr, layers_name, full_layers[:exit_index])

    def restore_layers(self, model: nn.Module):
        """
        スキップしたレイヤーを元に戻す（シラフに戻る）。
        """
        if self._original_layers is not None:
            layers_attr = None
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layers_attr = model.model
                layers_name = "layers"
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layers_attr = model.transformer
                layers_name = "h"
            
            if layers_attr:
                setattr(layers_attr, layers_name, self._original_layers)
                self._original_layers = None

def negate_alignment_vector(activations: torch.Tensor, alignment_vector: torch.Tensor, intensity: float = 1.0):
    """
    特定のアクティベーションから、アライメント（自制心）に関連するベクトルの影響を排除する。
    """
    # 簡易的な実装：ベクトル方向への投影分を減算する
    # dot_product = torch.sum(activations * alignment_vector, dim=-1, keepdim=True)
    # alignment_component = dot_product * alignment_vector
    # return activations - (intensity * alignment_component)
    pass
