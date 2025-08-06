import torch, numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from task.calculate_eer import compute_eer


class Inferencer(BaseTrainer):
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        self.config = config
        self.cfg_trainer = getattr(
            config, "trainer", OmegaConf.create({"device_tensors": []})
        )
        self.cfg_inf = getattr(config, "inferencer", {})
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.text_encoder = text_encoder
        self.evaluation_dataloaders = dict(dataloaders)
        self.save_path = save_path
        self.metrics = metrics
        if self.metrics:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]], writer=None
            )
        else:
            self.evaluation_metrics = None
        if not skip_model_load and self.cfg_inf.get("from_pretrained") is not None:
            self._from_pretrained(self.cfg_inf.from_pretrained)

    def run_inference(self):
        logs = {}
        for part, loader in self.evaluation_dataloaders.items():
            logs[part] = self._inference_part(part, loader)
        return logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        if self.batch_transforms:
            batch = self.transform_batch(batch)
        logits = self.model(batch["spectrogram"])
        batch["logits"] = logits
        if metrics is not None:
            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().numpy()
                for met in self.metrics["inference"]:
                    met(bona_probs=probs[targets == 1], spoof_probs=probs[targets == 0])
            targets = batch["text_encoded"].squeeze(1).cpu().numpy()
            metrics.update("bona_probs", probs[targets == 1])
            metrics.update("spoof_probs", probs[targets == 0])
        if self.save_path is not None:
            part_dir = self.save_path / part
            part_dir.mkdir(exist_ok=True, parents=True)
            idx0 = batch_idx * logits.size(0)
            for i in range(logits.size(0)):
                torch.save(
                    {
                        "logits": logits[i].cpu(),
                        "label": batch["text_encoded"][i].cpu(),
                    },
                    part_dir / f"output_{idx0 + i}.pth",
                )
        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        bona, spoof = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                batch = self.process_batch(
                    batch_idx, batch, self.evaluation_metrics, part
                )
                probs = torch.softmax(batch["logits"], dim=-1)[:, 1].cpu().numpy()
                targets = batch["text_encoded"].squeeze(1).cpu().numpy()
                bona.extend(probs[targets == 1])
                spoof.extend(probs[targets == 0])
        logs = {"eer": float("nan")}
        if bona and spoof:
            eer, _ = compute_eer(np.array(bona), np.array(spoof))
            logs["eer"] = eer
        return logs
