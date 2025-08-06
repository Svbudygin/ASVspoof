from src.trainer.base_trainer import BaseTrainer
from src.metrics.tracker import MetricTracker
import torch, numpy as np


class Trainer(BaseTrainer):

    def _log_batch(self, batch_idx, batch, mode="train"):
        self.writer.add_scalar(f"{mode}/loss", batch["loss"].item())

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        spectrogram = batch["spectrogram"]
        targets = batch["text_encoded"].squeeze(1)

        logits = self.model(spectrogram)
        loss = self.criterion(logits, targets)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            bona_mask = targets_np == 1
            spoof_mask = targets_np == 0

            batch.update(
                {
                    "bona_probs": probs_np[bona_mask],
                    "spoof_probs": probs_np[spoof_mask],
                }
            )

        metrics.update("loss", loss.item())
        if not self.is_train:
            for m in self.metrics["inference"]:
                m(bona_probs=probs[targets_np == 1], spoof_probs=probs[targets_np == 0])
        batch.update({"loss": loss})
        return batch
