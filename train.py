import os
import time
from datetime import datetime
import logging
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import random
from src.text_encoder import ASVTextEncoder


import torch.nn.functional as F
from typing import List, Tuple

def pad_collate(batch):
    feats, labels, keys = zip(*batch)             
    feats = torch.stack(feats).unsqueeze(1)       
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1)  
    lengths = torch.full((feats.size(0),), feats.size(-1), dtype=torch.long)

    return {
        "spectrogram": feats,   
        "text_encoded": labels,    
        "lengths": lengths,
        "utt_id": list(keys),
    }



def make_loader_asvspoof(root: str, split: str, batch_size: int, *, 
                         bonafide_ratio: float | None = None,
                         n_fft: int = 512, hop_length: int = 160, win_length: int = 400,
                         spec_aug_cfg: dict | None = None,
                         num_workers: int = 4, shuffle: bool | None = None):
    """
    Создаёт DataLoader для ASVspoof с LPS-фронтендом и фиксированной длиной сегмента.
    Параметры совпадают с тем, что вы прислали.
    """
    from hydra.utils import instantiate
    from src.datasets.asv import ASVspoofDataset
    ds_kwargs = dict(
        root=root,
        split=split,
        bonafide_ratio=bonafide_ratio,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
    )
    if spec_aug_cfg and split == "train":
        try:
            ds_kwargs["spec_aug"] = instantiate(spec_aug_cfg)
        except Exception:
            ds_kwargs["spec_aug"] = None
    dataset = ASVspoofDataset(**ds_kwargs)
    if shuffle is None:
        shuffle = (split == "train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=pad_collate, pin_memory=True)
    return loader


from src.model.light_cnn import LightCNN
from src.logger import CometMLWriter, py_logger, _ensure_logger_configured
from hydra.utils import to_absolute_path, instantiate
from src.trainer import Trainer
from src.metrics import EERMetric


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



@hydra.main(config_path="src/configs", config_name="asvspoof", version_base=None)
def main(cfg: DictConfig):
    """Training entrypoint for LightCNN on ASVspoof."""
    set_seed(cfg.train.seed)

    _ensure_logger_configured()

    train_loader = make_loader_asvspoof(
        root=to_absolute_path("data/ASVspoof2019_LA"),
        split="train",
        batch_size=getattr(cfg.train, "batch_size", 64),
        bonafide_ratio=0.3,
        n_fft=512, hop_length=160, win_length=400,
        spec_aug_cfg={
            "_target_": "src.datasets.spec_augment.SpecAugment",
            "freq_mask_param": 15,
            "time_mask_param": 40,
            "num_freq_masks": 2,
            "num_time_masks": 2,
            "p": 0.5,
        },
        num_workers=getattr(cfg, "num_workers", 4),
        shuffle=True,
    )
    val_loader = make_loader_asvspoof(
        root=to_absolute_path("data/ASVspoof2019_LA"),
        split="dev",
        batch_size=getattr(cfg.dev, "batch_size", getattr(cfg.train, "batch_size", 64)),
        bonafide_ratio=None,
        n_fft=512, hop_length=160, win_length=400,
        spec_aug_cfg=None,
        num_workers=getattr(cfg, "num_workers", 4),
        shuffle=False,
    )
    eval_root = to_absolute_path("data/ASVspoof2019_LA")
    eval_loader = make_loader_asvspoof(
        root=eval_root,
        split="eval",
        batch_size=getattr(cfg.eval, "batch_size", getattr(cfg.train, "batch_size", 64)),
        bonafide_ratio=None,
        n_fft=512, hop_length=160, win_length=400,
        spec_aug_cfg=None,
        num_workers=getattr(cfg, "num_workers", 4),
        shuffle=False,
    )

    model = LightCNN().to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    comet = CometMLWriter(
        logger=py_logger,
        project_config=cfg,
        project_name=cfg.logger.project_name,
        workspace=cfg.logger.workspace,
        run_name=cfg.logger.run_name,
        mode=cfg.logger.mode,
        loss_names=cfg.logger.loss_names,
        log_checkpoints=cfg.logger.log_checkpoints,
        id_length=cfg.logger.id_length,
        api_key=cfg.logger.api_key,
    )

    metrics = {"train": [], "inference": [EERMetric()], "dev": [EERMetric()]}

    save_dir = Path(to_absolute_path(cfg.trainer.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    text_encoder = ASVTextEncoder()

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        text_encoder=text_encoder,
        config=cfg,
        device=cfg.device,
        dataloaders={"train": train_loader, "dev": val_loader, "eval": eval_loader},
        logger=py_logger,
        writer=comet,
    )
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.train()


if __name__ == "__main__":
    main()
