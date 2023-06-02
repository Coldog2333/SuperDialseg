import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional, Union
import torch
from transformers import PreTrainedTokenizer

from .utils.data.load_dialseg import DialsegSample
from .utils.data import ROLE2LABEL, DOC2DIAL_DA2LABEL


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


def call_tokenizer(
    tokenizer: PreTrainedTokenizer,
    tokens: List[str],
    add_special_tokens=False,
    padding: Optional[str] = 'max_length',
    max_length=512,
    truncation=True,
    return_attention_mask=True,
    return_tensor='pt'
) -> Dict[str, Union[List, torch.Tensor]]:
    encodings = defaultdict(list)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # first truncate
    if truncation:
        input_ids = input_ids[:max_length]

    attention_mask = [1] * len(input_ids)

    # then pad
    if padding:
        origin_length = len(input_ids)
        input_ids += [tokenizer.pad_token_id] * (max_length - origin_length)
        attention_mask += [0] * (max_length - origin_length)
        assert (len(input_ids) == len(attention_mask) == max_length)

    # make return values
    encodings['input_ids'] = input_ids
    if return_attention_mask:
        encodings['attention_mask'] = attention_mask

    if return_tensor == 'pt':
        for key in encodings.keys():
            encodings[key] = torch.tensor(encodings[key], dtype=torch.long)

    return encodings


@dataclass
class DataCollatorForSupervisedDialogueSegmentation(object):  # TODO: Building
    """Collate examples for supervised fine-tuning."""

    args: None
    mode: str
    tokenizer: PreTrainedTokenizer

    def __getitem_input__(self, sample: DialsegSample, padding=None) -> Dict[str, torch.Tensor]:
        # set segmentation_label=1 for the last utterance to train better
        # but remember to recover it when doing evaluation

        ## make input text
        input_tokens = []
        classification_mask = []

        if self.args.backbone in ['TODBERT/TOD-BERT-JNT-V1']:
            input_tokens += [self.tokenizer.cls_token]
            classification_mask += [0]

            todbert_role_token_mapper = {'agent': '[sys]', 'user': '[usr]'}
            for i in range(len(sample)):
                tokens = self.tokenizer.tokenize(sample.utterances[i])[:self.args.max_utterance_len - 1]
                input_tokens += [todbert_role_token_mapper[sample.roles[i]]] + tokens
                classification_mask += [1] + [0] * (len(tokens) - 1)

            input_tokens += [self.tokenizer.sep_token]
            classification_mask += [0]

        elif self.args.backbone in ['bert-base-uncased', 'bert-large-uncased']:
            input_tokens += [self.tokenizer.cls_token]
            classification_mask += [0]

            for i in range(len(sample)):
                tokens = self.tokenizer.tokenize(sample.utterances[i])[:self.args.max_utterance_len - 1]  # limit the length of an utterance.
                input_tokens += tokens + ['[SEP]']
                classification_mask += [0] * len(tokens) + [1]

        elif self.args.backbone in ['roberta-base', 'roberta-large']:
            input_tokens += [self.tokenizer.cls_token]
            classification_mask += [0]

            for i in range(len(sample)):
                max_utterance_len = self.args.max_utterance_len - 3 if self.args.use_mask else self.args.max_utterance_len - 2
                tokens = self.tokenizer.tokenize(sample.utterances[i])[:max_utterance_len]  # limit the length of an utterance.
                input_tokens += tokens
                classification_mask += [0] * len(tokens)

                if self.args.use_mask:
                    input_tokens += ['<mask>', '</s>', '</s>']
                    classification_mask += [1, 0, 0]
                else:
                    input_tokens += ['</s>', '</s>']
                    classification_mask += [1, 0]

            input_tokens = input_tokens[:-1]
            classification_mask = classification_mask[:-1]

        elif self.args.backbone in ['t5-base', 't5-large']:
            for i in range(len(sample)):
                tokens = self.tokenizer.tokenize(sample.utterances[i])[:self.args.max_utterance_len - 1]  # limit the length of an utterance.
                input_tokens += tokens + ['</s>']
                classification_mask += [0] * len(tokens) + [1]

        else:
            raise ValueError('Unsupported backbone: [%s]' % self.args.backbone)

        # make inputs
        encodings = call_tokenizer(
            self.tokenizer,
            tokens=input_tokens,
            add_special_tokens=False,
            padding=padding,
            max_length=self.args.max_seq_len,
            truncation=True,
            return_attention_mask=False,
            return_tensor='pt'
        )
        input_ids = encodings['input_ids'].squeeze()

        ## make labels and other annotations
        # TODO: Cannot do that!!!
        # labels, da_labels, role_labels = [torch.ones_like(input_ids, dtype=torch.long) * -100] * 3
        labels, da_labels, role_labels = [torch.ones_like(input_ids, dtype=torch.long) * -100 for _ in range(3)]
        da_ids, role_ids = [torch.zeros_like(input_ids, dtype=torch.long) for _ in range(2)]

        if self.mode == 'train':
            # Don't consider the last segmentation point
            sample.segmentation_labels[-1] = -100

        i = 0
        for j, mask in enumerate(classification_mask):
            # avoid overflow
            not_overflow_i = min(i, len(sample) - 1)
            da_ids[j] = DOC2DIAL_DA2LABEL[sample.dialogue_acts[not_overflow_i]]
            role_ids[j] = ROLE2LABEL[sample.roles[not_overflow_i]]
            if mask:
                labels[j] = sample.segmentation_labels[i]
                da_labels[j] = DOC2DIAL_DA2LABEL[sample.dialogue_acts[i]]
                role_labels[j] = ROLE2LABEL[sample.roles[i]]
                i += 1

        return {
            'input_ids': input_ids,
            'labels': labels,
            'da_labels': da_labels,
            'role_labels': role_labels,
            'da_ids': da_ids,
            'role_ids': role_ids
        }

    def __getitem__(self, sample) -> Dict[str, torch.Tensor]:
        if self.mode in ['test', 'validation']:
            all_inputs = defaultdict(list)

            for i in range(max(1, len(sample) - self.args.sliding_window + 1)):
                current_sample = sample[i: i + self.args.sliding_window]
                inputs = self.__getitem_input__(current_sample, padding='max_length')

                for key, value in inputs.items():
                    all_inputs[key].append(value)

            for key in all_inputs.keys():
                all_inputs[key] = torch.stack(all_inputs[key])

            return all_inputs

        else:
            if len(sample) > self.args.sliding_window:
                # static truncation
                # start_index = 0
                # sliding window
                start_index = random.randint(0, len(sample) - self.args.sliding_window)
                sample = sample[start_index: start_index + self.args.sliding_window - 1]

            return self.__getitem_input__(sample)

    def __call__(self, samples: Sequence[DialsegSample]) -> Dict[str, torch.Tensor]:
        input_dicts = [self.__getitem__(sample) for sample in samples]

        input_ids, labels, da_labels, role_labels, da_ids, role_ids = tuple(
            [input_dict[key] for input_dict in input_dicts] for key in
            ("input_ids", "labels", "da_labels", "role_labels", "da_ids", "role_ids")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        da_labels = torch.nn.utils.rnn.pad_sequence(da_labels, batch_first=True, padding_value=IGNORE_INDEX)
        role_labels = torch.nn.utils.rnn.pad_sequence(role_labels, batch_first=True, padding_value=IGNORE_INDEX)
        da_ids = torch.nn.utils.rnn.pad_sequence(da_ids, batch_first=True, padding_value=0)
        role_ids = torch.nn.utils.rnn.pad_sequence(role_ids, batch_first=True, padding_value=0)

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            da_labels=da_labels,
            role_labels=role_labels,
            da_ids=da_ids,
            role_ids=role_ids
        )
