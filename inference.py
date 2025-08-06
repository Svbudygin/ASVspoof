import os
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hydra.utils import to_absolute_path, instantiate

from src.model.light_cnn import LightCNN
from src.logger import py_logger, _ensure_logger_configured
from src.trainer import Inferencer
from src.text_encoder import ASVTextEncoder
from src.datasets.asv import ASVspoofDataset   


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pad_collate(batch):
    feats, labels, names = zip(*batch)          
    feats = torch.stack(feats).unsqueeze(1)     
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1)  
    lengths = torch.full((feats.size(0),), feats.size(-1), dtype=torch.long)
    return {
        "spectrogram": feats,
        "text_encoded": labels,    
        "lengths": lengths,
        "utt_id": list(names),
    }

def make_loader_asvspoof(
    root: str,
    split: str,                      
    batch_size: int = 64,
    *,
    bonafide_ratio: float | None = None,    
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    spec_aug_cfg: dict | None = None,       
    num_workers: int = 4,
    shuffle: bool = False,                   
):
    ds_kwargs = dict(
        root=root,
        split=split,
        bonafide_ratio=bonafide_ratio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    if spec_aug_cfg and split == "train":
        try:
            ds_kwargs["spec_aug"] = instantiate(spec_aug_cfg)
        except Exception:
            ds_kwargs["spec_aug"] = None

    dataset = ASVspoofDataset(**ds_kwargs)
    py_logger.info("[ASVspoofDataset] %s: %d wavs (bona_ratio=%s)", split, len(dataset), str(bonafide_ratio))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    return loader


@hydra.main(config_path="src/configs", config_name="asvspoof", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    _ensure_logger_configured()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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

    model = LightCNN().to(device)
    model.eval()

    ckpt_path = (
        getattr(cfg, "inferencer", {}).get("from_pretrained", None)
        or getattr(cfg, "inference", {}).get("from_pretrained", None)
        or "checkpoints/model_best.pth"      
    )
    ckpt_path = to_absolute_path(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    py_logger.info("Loading model weights from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt.get("state_dict")
        or ckpt.get("model_state")
        or ckpt.get("model_state_dict")
        or ckpt
    )
    model.load_state_dict(state_dict, strict=True)

    text_encoder = ASVTextEncoder()  
    inferencer = Inferencer(
        model=model,
        config=cfg,
        device=device,
        dataloaders={"eval": eval_loader},
        text_encoder=text_encoder,
        save_path=None,
        metrics=None,
        batch_transforms=None,
        skip_model_load=True,   
    )


    eval_logs = inferencer.run_inference()
    final_eer = eval_logs.get("eval", {}).get("eer", float("nan"))
    py_logger.info("Final honest EER on evaluation set: %.2f%%", final_eer * 100)


    # from pathlib import Path
    out_csv = Path(to_absolute_path("students_solutions/svbudygin.csv"))
    with torch.inference_mode():
        for batch in eval_loader:
            specs = batch["spectrogram"].to(device)
            logits = model(specs)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(bonafide)
            names = batch["utt_id"]
            with out_csv.open("a") as f:
                for name, p in zip(names, probs):
                    f.write(f"{name},{p:.6f}\n")
    py_logger.info("Saved scores to %s", str(out_csv))


if __name__ == "__main__":
    main()
