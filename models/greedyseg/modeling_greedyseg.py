import torch
import argparse
import numpy as np
import os
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertModel

from models.greedyseg.utils_greedyseg import convert_examples_to_features, read_expamples_2
from modeling_utils import BaseSegmenter

WINDOW_SIZE = 2
SEGMENT_JUMP_STEP = 2
SIMILARITY_THRESHOLD = 0.6
MAX_SEGMENT_ROUND = 8   # original code: 6
MAX_SEQ_LENGTH = 50
MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
}


def similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


class GreedySegmenter(BaseSegmenter):
    def __init__(
        self,
        backbone: str = 'bert-base-uncased',
        max_utterance_len: int = MAX_SEQ_LENGTH,
    ):
        super().__init__()

        self.backbone = backbone
        self.max_utterance_len = max_utterance_len

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.backbone)
        self.model = BertModel.from_pretrained(self.backbone).to(self.device)
        self.model.eval()

    def generate_vectors(self, examples):
        features = convert_examples_to_features(
            examples, self.max_utterance_len, self.tokenizer,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
        )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=all_input_ids,
                attention_mask=all_input_mask,
                token_type_ids=all_segment_ids,
                return_dict=True
            )

        vectors = outputs['last_hidden_state'][:, 0, :]
        return vectors.cpu().detach().numpy()

    def forward(self, inputs):
        document = inputs['utterances']

        if len(document) % 2:
            document = document[1:]

        cut_index = 0
        cut_list = []
        while cut_index < len(document):
            left_sent = ""
            i = 0
            temp_sent = ""
            final_value = 2
            final_cutpoint = len(document) - 1

            if cut_index - WINDOW_SIZE > 0:
                index = WINDOW_SIZE
                while index > 0:
                    left_sent += document[cut_index - index]
                    index -= 1

            else:
                temp_index = 0
                while temp_index < cut_index:
                    left_sent += document[temp_index]
                    temp_index += 1

            while cut_index + i < len(document) and i < MAX_SEGMENT_ROUND:
                temp_sent += document[cut_index + i]
                # 加在判断后面 or (i==0 and cut_index + i == len(document)-1) 此时final_cutpoint可以随意初始化 如果剩下一句话满足不了进入判断要求
                if i % SEGMENT_JUMP_STEP == SEGMENT_JUMP_STEP - 1:
                    bert_input = []
                    right_sent = ""
                    if cut_index + i + WINDOW_SIZE < len(document):
                        index = 1
                        while (index <= WINDOW_SIZE):
                            right_sent += document[cut_index + i + index]
                            index += 1
                    else:
                        temp_index = 1
                        while (cut_index + i + temp_index < len(document)):
                            right_sent += document[cut_index + i + temp_index]
                            temp_index += 1

                    if left_sent:
                        bert_input.append(left_sent)
                    bert_input.append(temp_sent)
                    if right_sent:
                        bert_input.append(right_sent)
                    examples = read_expamples_2(bert_input)
                    vectors = self.generate_vectors(examples)
                    if left_sent:
                        left_value = similarity(vectors[0], vectors[1])
                        right_value = similarity(vectors[1], vectors[2]) if right_sent else -1
                    else:
                        left_value = -1
                        right_value = similarity(vectors[0], vectors[1]) if right_sent else -1
                    larger_value = left_value if left_value > right_value else right_value
                    if not left_sent and not right_sent:  # 防止前后都没有参考窗口，即len(document)<=MAX_SEGMENT_ROUND

                        larger_value = SIMILARITY_THRESHOLD
                        # 如果中间截断的情况的最小相似性都大于0.8则这段通话不进行切分,
                        # 中间截断的情况只有小于这个阈值才会截断，
                    if larger_value < final_value:
                        final_value = larger_value
                        final_cutpoint = cut_index + i
                i += 1

            cut_list.append(final_cutpoint)
            cut_index = final_cutpoint + 1

        if len(inputs['utterances']) % 2:
            cut_list_new = [i + 1 for i in cut_list]
        else:
            cut_list_new = cut_list
        if len(cut_list) == 0:
            cut_list_new = [0]

        # ---------------------------dcz
        predictions = [0] * len(inputs['utterances'])
        for i in cut_list_new:
            predictions[i] = 1
        predictions[-1] = 1
        # ---------------------------dcz
        assert (len(inputs['utterances']) == len(predictions))

        return predictions
