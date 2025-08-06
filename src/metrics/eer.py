import numpy as np
from task.calculate_eer import compute_eer
from src.metrics.base_metric import BaseMetric
import torch


class EERMetric(BaseMetric):
    name = "eer"

    def __init__(self):
        self.reset()

    def reset(self):
        self.bona, self.spoof = [], []

    def __call__(self, *, bona_probs=None, spoof_probs=None, **_):
        def to_np(x):
            if x is None:
                return np.empty(0)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        self.bona.extend(to_np(bona_probs))
        self.spoof.extend(to_np(spoof_probs))
        return None

    def compute(self):
        if not self.bona or not self.spoof:
            return float("nan")
        eer, _ = compute_eer(np.array(self.bona), np.array(self.spoof))
        return eer
