import torch
from torch import nn, Tensor


class CrossEntropyLossWrapper(nn.Module):
    """Wrapper over :class:`torch.nn.CrossEntropyLoss` returning a dict.

    The wrapper expects ``logits`` and ``text_encoded`` keys in the batch and
    returns a dictionary with a single ``loss`` key so it fits into the existing
    training loop.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits: Tensor, text_encoded: Tensor, **batch) -> dict:
        targets = text_encoded.squeeze(1)
        loss = self.loss_fn(logits, targets)
        return {"loss": loss}


class ASoftmaxLoss(nn.Module):
    """Simple additive-margin (A-Softmax) loss.

    The implementation applies an additive margin to the logits of the target
    class before scaling and computing cross-entropy. It is intentionally kept
    lightweight so that different margin/scale configurations can be explored
    later.
    """

    def __init__(self, margin: float = 0.35, scale: float = 30.0, **kwargs):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits: Tensor, text_encoded: Tensor, **batch) -> dict:
        targets = text_encoded.squeeze(1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        adjusted_logits = self.scale * (logits - self.margin * one_hot)
        loss = self.ce(adjusted_logits, targets)
        return {"loss": loss}
