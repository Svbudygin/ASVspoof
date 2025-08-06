import torch


class ASVTextEncoder:
    mapping = {"spoof": 0, "bonafide": 1}

    def __len__(self):
        return len(self.mapping)

    def encode(self, text: str) -> torch.Tensor:
        idx = self.mapping[text.lower()]
        return torch.tensor([[idx]], dtype=torch.long)
