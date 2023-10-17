from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence
import gensim
import torch
from ...utils.data.load_dialseg import DialsegSample
from .tokenization_textseg import TextsegTokenizer


@dataclass
class DataCollatorForTextseg(object):
    word2vec: Optional[gensim.models.KeyedVectors] = None

    def __post_init__(self):
        self.tokenizer = TextsegTokenizer(word2vec=self.word2vec)

    def __call__(self, samples: Sequence[DialsegSample]) -> Dict[str, list]:
        batch_input_ids, batch_labels, batch_n_sentence, batch_utterance_lengths = [], [], [], []
        for sample in samples:
            n_sentence = len(sample)
            for sentence in sample.utterances:
                tokens = self.tokenizer.tokenize(sentence, remove_special_tokens=True)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                assert (len(input_ids) >= 1), sentence

                input_ids = torch.tensor(input_ids, dtype=torch.long)

                batch_input_ids.append(input_ids)
                batch_utterance_lengths.append(len(input_ids))

            batch_labels.extend(sample.segmentation_labels[:-1])
            batch_n_sentence.append(n_sentence)

        # padding
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            'input_ids': batch_input_ids,
            'labels': batch_labels,
            'dialogue_length': batch_n_sentence,
            'utterance_length': batch_utterance_lengths
        }
