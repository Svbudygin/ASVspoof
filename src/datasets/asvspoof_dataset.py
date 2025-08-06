from pathlib import Path
from typing import List

import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    """Dataset for ASVspoof data from protocol files.

    Parameters
    ----------
    protocol_path: str or Path
        Path to the protocol file describing the dataset split.
    audio_root: str or Path
        Root directory containing audio files referenced in the protocol.
    *args, **kwargs: passed to :class:`BaseDataset`.
    """

    def __init__(self, protocol_path, audio_root, *args, **kwargs):
        index = self._create_index(protocol_path, audio_root)
        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path, audio_root) -> List[dict]:
        protocol_path = Path(protocol_path)
        audio_root = Path(audio_root)

        index = []
        if not protocol_path.exists():
            return index

        with protocol_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                audio_token = parts[1]
                if Path(audio_token).suffix == "":
                    audio_token = f"{audio_token}.flac"

                label = next(
                    (p.lower() for p in parts if p.lower() in ["bonafide", "spoof"]),
                    "",
                )

                path = (audio_root / audio_token).expanduser().resolve()

                audio_len = 0.0
                if path.exists():
                    t_info = torchaudio.info(str(path))
                    audio_len = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {"path": str(path), "text": label, "audio_len": audio_len}
                    )
        return index
