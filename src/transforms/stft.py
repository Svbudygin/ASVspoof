import torchaudio
import torch
from torch import nn

class LPSFrontend(nn.Module):
    def __init__(self, n_fft=512, win_length=400, hop_length=160):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            power=2.0, center=True
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, wav: torch.Tensor):        # (1, T)
        S = self.spec(wav)                       # (1, F, T)
        S = self.to_db(S)                        # dB
        return S.squeeze(0)                      # (F, T) — 1 канал «как у картинки»