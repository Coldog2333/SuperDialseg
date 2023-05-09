from dataclasses import dataclass
from typing import Dict, Sequence
import torch

from utils.data.load_dialseg import DialsegSample


IGNORE_INDEX = -100


@dataclass
class DataCollatorForSegmentationLabels(object):
    def __call__(self, samples: Sequence[DialsegSample]) -> Dict[str, torch.Tensor]:
        labels = [torch.tensor(sample.segmentation_labels) for sample in samples]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(labels=labels)


@dataclass
class DataCollatorPlainText(object):
    def __call__(self, samples: Sequence[DialsegSample]) -> Dict[str, torch.Tensor]:
        labels = [torch.tensor(sample.segmentation_labels) for sample in samples]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        utterances = [sample.utterances for sample in samples]
        return dict(utterances=utterances, labels=labels)
