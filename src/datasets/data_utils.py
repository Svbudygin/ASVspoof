from itertools import repeat

from hydra.utils import instantiate
from src.transforms.stft import LPSFrontend
from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
from src.datasets.asvspoof_dataset import ASVspoofDataset
from torch.utils.data import DataLoader
from src.text_encoder import ASVTextEncoder
from src.logger import py_logger


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def make_loader(protocol, audio_root, batch_size=16):
    py_logger.info(
        "Creating DataLoader from:\n - protocol: %s\n - audio_root: %s",
        protocol,
        audio_root,
    )
    text_encoder = ASVTextEncoder()
    transforms = {"get_spectrogram": LPSFrontend()}
    dataset = ASVspoofDataset(
        protocol_path=protocol,
        audio_root=audio_root,
        text_encoder=text_encoder,
        instance_transforms=transforms,
    )
    py_logger.info("Dataset loaded with %d samples.", len(dataset))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return loader, text_encoder
