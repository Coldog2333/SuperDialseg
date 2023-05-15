import json
import warnings
from dataclasses import dataclass
from typing import List, Dict

from torch.utils.data import Dataset


@dataclass
class DialsegSample:
    utterances: List[str]
    segmentation_labels: List[int] = None  # set 1 for the end of a topic, except for the last utterance -> last one must be 0.
    roles: List[str] = None                # Default: system,user,system,user,...
    dialogue_acts: List[str] = None        # If possible
    topics: List[str] = None               # If possible
    sources: List[str] = None              # If possible

    def __post_init__(self):
        # unit check
        if self.segmentation_labels is not None:
            assert (len(self.utterances) == len(self.segmentation_labels)), \
                "The number of utterances and segmentation labels should be the same."

            # set seg_label=1 for the last utterance
            # if self.segmentation_labels[-1] != 1:
            #     warnings.warn(message='Segmentation Label for the last utterance should be 1. It is reset as 1. '
            #                           'But remember to reset it back to 0 when evaluating.')
            #     self.segmentation_labels[-1] = 1
            # set seg_label=0 for the last utterance
            self.segmentation_labels[-1] = 0

        # default
        ## Default roles: system, user, system, user, ...
        if self.roles is None:
            roles = ['system', 'user'] * round(len(self.utterances) / 2)
            self.roles = roles[:self.__len__()]

        ## Default dialogue_acts: "", "", "", ...
        if self.dialogue_acts is None:
            self.dialogue_acts = [""] * self.__len__()

    def __getitem__(self, index):
        return DialsegSample(
            utterances=self.utterances[index],
            segmentation_labels=self.segmentation_labels[index] if self.segmentation_labels is not None else None,
            roles=self.roles[index] if self.roles is not None else None,
            dialogue_acts=self.dialogue_acts[index] if self.dialogue_acts is not None else None,
            topics=self.topics[index] if self.topics is not None else None,
            sources=self.sources[index] if self.sources is not None else None,
        )

    def __len__(self):
        return len(self.utterances)

    def to_json(self) -> Dict:
        json_data = {
            'dial_id': '',
            'turns': [],
        }
        topic_id = 0
        for i in range(self.__len__()):
            turn = {
                'dialogue_act': self.dialogue_acts[i],
                'role': self.roles[i],
                'utterance': self.utterances[i],
                'topic_id': topic_id,
                'segmentation_label': self.segmentation_labels[i],
                'topic': self.topics[i],
                'source': self.sources[i]
            }
            json_data['turns'].append(turn)

            if self.segmentation_labels[i] == 1:
                topic_id += 1
        return json_data


def load_segmentation_samples_from_json(segmentation_file) -> List[DialsegSample]:
    """
        :param segmentation_json_file: absolute path of segmenation json file
            example:
        :return:
            segmentation_samples: list of <DialsegSample>
    """
    dial_data = json.load(open(segmentation_file, encoding='utf-8'))['dial_data']
    dataset_name = list(dial_data.keys())[0]
    dialogues = dial_data[dataset_name]

    segmentation_samples = []
    for dialogue in dialogues:
        turns = dialogue['turns']

        utterances = [turn['utterance'] for turn in turns]
        segmentation_labels = [turn['segmentation_label'] for turn in turns]
        roles = [turn['role'] for turn in turns] if 'role' in turns[0].keys() else None
        dialogue_acts = [turn['dialogue_act'] for turn in turns] if 'dialogue_act' in turns[0].keys() else None

        dialseg_sample = DialsegSample(
            utterances=utterances,
            segmentation_labels=segmentation_labels,
            roles=roles,
            dialogue_acts=dialogue_acts,
        )

        segmentation_samples.append(dialseg_sample)

    return segmentation_samples


def load_segmentation_samples_from_txt(segmentation_file) -> List[DialsegSample]:
    """
    :param segmentation_file: absolute path of segmenation file
        example:
            call_id_0
            0 Los Angeles, please. Will it be hot?
            0 It will be hot today in Los Angeles.
            1 Yes, can you give me the information on the Huntingdon Marriott Hotel?
            1 Absolutely. It is an expensive hotel located in the west part of town. It has 4 starts and includes free wifi and parking. Would you like help booking a room?
            call_id_1
            0 You are welcome.
            1 what will the weather be like in the city
            ...
    :return:
        segmentation_samples: list of <DialsegSample>
    """
    with open(segmentation_file, encoding='utf-8') as f:
        segmentation_samples = []
        utterances, segmentation_labels = [], []

        f.readline()  # skip the first line
        previous_topic_id = -1
        for line in f:
            if line.startswith('call_id'):
                segmentation_labels[-1] = 1    # but remember to set as 0 when doing evaluation.
                segmentation_samples.append(
                    DialsegSample(
                        utterances=utterances,
                        segmentation_labels=segmentation_labels
                    )
                )
                previous_topic_id, utterances, segmentation_labels = 0, [], []
            else:
                utterances.append(' '.join(line.strip().split(' ')[1:]))
                current_topic_id = int(line.split(' ')[0])

                if current_topic_id != previous_topic_id and segmentation_labels != []:
                    segmentation_labels[-1] = 1
                segmentation_labels.append(0)
                previous_topic_id = current_topic_id

        # for the last one
        segmentation_labels[-1] = 1  # but remember to set as 0 when doing evaluation.
        segmentation_samples.append(
            DialsegSample(
                utterances=utterances,
                segmentation_labels=segmentation_labels
            )
        )
        previous_topic_id, utterances, segmentation_labels = 0, [], []  # for engineering safety

    return segmentation_samples


class DialsegDataset(Dataset):
    def __init__(self, segmentation_file):
        super(DialsegDataset, self).__init__()
        self.segmentation_file = segmentation_file

        if self.segmentation_file.endswith('.json'):
            self.samples = load_segmentation_samples_from_json(segmentation_file)
        elif self.segmentation_file.endswith('.txt'):
            self.samples = load_segmentation_samples_from_txt(segmentation_file)
        else:
            raise NotImplementedError('Only support *.json and *.txt files.')

    def __getitem__(self, index) -> DialsegSample:
        sample = self.samples[index]
        return sample

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == '__main__':
    segmentation_file = './segmentation_file_validation.json'
    dataset = DialsegDataset(segmentation_file)
    print(dataset[0].to_json())
