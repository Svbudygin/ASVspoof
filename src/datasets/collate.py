import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """Create batch from list of dataset items.

    Each item in ``dataset_items`` is expected to be a dictionary produced by
    :func:`BaseDataset.__getitem__` and contain at least the following keys:
    ``audio`` (Tensor), ``spectrogram`` (Tensor) and ``text_encoded`` (Tensor).

    The function pads variable-length tensors in the batch to the length of the
    longest element using :func:`torch.nn.utils.rnn.pad_sequence` and also
    computes their original lengths. The resulting batch dictionary matches the
    expectations of the model and the CTC criterion.

    Args:
        dataset_items (list[dict]): list of objects from ``dataset.__getitem__``.

    Returns:
        dict: batch with padded tensors and corresponding lengths.
    """

    audios = [item["audio"].squeeze(0) for item in dataset_items]
    spectrograms = [item["spectrogram"].transpose(0, 1) for item in dataset_items]
    labels = [item["text_encoded"].squeeze(0).long() for item in dataset_items]

    texts = [item.get("text") for item in dataset_items]
    audio_paths = [item.get("audio_path") for item in dataset_items]

    audio_lengths = torch.tensor([audio.shape[0] for audio in audios], dtype=torch.long)
    spectrogram_lengths = torch.tensor(
        [spec.shape[0] for spec in spectrograms], dtype=torch.long
    )
    label_lengths = torch.tensor([label.shape[0] for label in labels], dtype=torch.long)

    audios = pad_sequence(audios, batch_first=True)
    spectrograms = (
        pad_sequence(spectrograms, batch_first=True).transpose(1, 2).unsqueeze(1)
    )
    labels = pad_sequence(labels, batch_first=True)

    result_batch = {
        "audio": audios,
        "audio_length": audio_lengths,
        "spectrogram": spectrograms,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": labels,
        "text_encoded_length": label_lengths,
        "text": texts,
        "audio_path": audio_paths,
    }

    return result_batch
