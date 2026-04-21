import torch
from transformers import LogitsProcessor

class DrunkenLogitsProcessor(LogitsProcessor):
    """
    ロジットにノイズを加え、特定の単語への確信度を揺らすことで
    「ろれつの回らなさ」や「とりとめのない思考」を再現するプロセッサ。
    """
    def __init__(self, intoxication_level: float = 0.5, slur_intensity: float = 1.0):
        self.level = intoxication_level
        self.slur_intensity = slur_intensity

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # ロジットにランダムなガウスノイズを加える
        # 酔い度が高いほど、本来選ばれるべき単語以外の確率が相対的に上がり、支離滅裂になる
        if self.level > 0:
            noise_scale = self.level * self.slur_intensity
            noise = torch.randn_like(scores, device=scores.device) * noise_scale
            return scores + noise
        return scores
